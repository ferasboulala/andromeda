#include <arm_neon.h>
#include <raylib.h>

#include <random>
#include <thread>

#include "spatial-tree.h"

static constexpr int Rank = 2;
static constexpr int MaximumLeafSize = 64;
using CoordinateType = float;
using MassType = float;

struct n_body_data {
    std::array<CoordinateType, Rank> acceleration;
    std::array<CoordinateType, Rank> velocity;
    std::array<CoordinateType, Rank> position;
    MassType                         mass;
};

struct n_body_tree_data_external {
    MassType                         mass;
    std::array<CoordinateType, Rank> position;
};

struct n_body_tree_data_internal {
    MassType mass;
    uint32_t idx;
};

struct n_body_tree_leaf_simd {
    uint32_t                                                      size;
    std::array<std::array<CoordinateType, MaximumLeafSize>, Rank> positions;
    std::array<MassType, MaximumLeafSize>                         masses;
};

using spatial_tree_type = st::internal::
    spatial_tree<CoordinateType, n_body_tree_data_internal, Rank, MaximumLeafSize, 32, true>;
class barnes_hut : public spatial_tree_type {
public:
    barnes_hut() = default;
    ~barnes_hut() = default;

    void build(const auto& points) {
        st::bounding_box<CoordinateType, Rank> boundary;
        st::bounding_box<CoordinateType, Rank> boundary_global;
        boundary_global.stops.fill(std::numeric_limits<CoordinateType>::lowest() / 2);
        boundary_global.starts.fill(std::numeric_limits<CoordinateType>::max() / 2);
#pragma omp parallel private(boundary)
        {
            boundary.stops.fill(std::numeric_limits<CoordinateType>::lowest() / 2);
            boundary.starts.fill(std::numeric_limits<CoordinateType>::max() / 2);
#pragma omp parallel for
            for (const auto& point : points) {
                assert(point.mass);
                st::internal::unroll_for<Rank>([&](auto i) {
                    boundary.starts[i] = std::min(point.position[i], boundary.starts[i]);
                    boundary.stops[i] = std::max(point.position[i], boundary.stops[i]);
                });
            }
#pragma omp critical
            st::internal::unroll_for<Rank>([&](auto i) {
                boundary_global.starts[i] = std::min(boundary_global.starts[i], boundary.starts[i]);
                boundary_global.stops[i] = std::max(boundary_global.stops[i], boundary.stops[i]);
            });
        }

        this->reset(boundary_global);
        uint32_t i = 0;
        for (const auto& point : points) {
            this->emplace(point.position, n_body_tree_data_internal{point.mass, i++});
        }
    }

    void propagate() {
        branch_data_.resize(this->branches_.size());
        for (int64_t i = this->branches_.size() - 1; i >= 0; --i) {
            const auto& branch = this->branches_[i];
            auto&       branch_data_entry = branch_data_[i];
            branch_data_entry.mass = 0;
            branch_data_entry.position = {0, 0};

            if (branch.is_terminal()) {
                auto& leaf = this->leaves_[branch.index()];
                std::for_each(leaf.items.begin(),
                              leaf.items.begin() + leaf.size,
                              [&](const auto& data) { branch_data_entry.mass += data.data.mass; });
                if (branch_data_entry.mass == 0.0) {
                    continue;
                }

                const MassType reciprocal = 1.0 / branch_data_entry.mass;
                for (uint64_t i = 0; i < leaf.size; ++i) {
                    const MassType weight = leaf.items[i].data.mass * reciprocal;
                    st::internal::unroll_for<Rank>([&](auto j) {
                        branch_data_entry.position[j] += leaf.coordinates[i][j] * weight;
                    });
                }
            } else {
                st::internal::unroll_for<spatial_tree_type::BranchingFactor>([&](auto i) {
                    const auto& child_branch_data_entry =
                        branch_data_[branch.index_of_first_child + i];
                    branch_data_entry.mass += child_branch_data_entry.mass;
                });
                if (branch_data_entry.mass == 0.0) continue;

                const MassType reciprocal = 1.0 / branch_data_entry.mass;
                st::internal::unroll_for<spatial_tree_type::BranchingFactor>([&](auto i) {
                    const auto& child_branch_data_entry =
                        branch_data_[branch.index_of_first_child + i];
                    st::internal::unroll_for<Rank>([&](auto j) {
                        branch_data_entry.position[j] += child_branch_data_entry.position[j] *
                                                         child_branch_data_entry.mass * reciprocal;
                    });
                });
            }
        }
    }

    void sort_and_vectorize(auto& points) {
        leaf_data_.resize(this->leaves_.size());
        std::vector<n_body_data> sorted(points.size());
        for (uint32_t i = 0, j = 0; i < this->leaves_.size(); ++i) {
            const auto& leaf = this->leaves_[i];
            auto&       leaf_data = leaf_data_[i];
            leaf_data.size = leaf.size;

            for (uint32_t k = 0; k < leaf.size; ++k, ++j) {
                sorted[j] = points[leaf.items[k].data.idx];
                leaf_data.masses[k] = sorted[j].mass;
                st::internal::unroll_for<Rank>(
                    [&](auto l) { leaf_data.positions[l][k] = sorted[j].position[l]; });
            }

            for (uint32_t k = leaf.size; k < MaximumLeafSize; ++k) {
                leaf_data.masses[k] = 0;
            }
        }

        points = std::move(sorted);
    }

    static inline float reduce_sum(float32x4_t vec) {
        float32x2_t sum_pair = vadd_f32(vget_low_f32(vec), vget_high_f32(vec));
        return vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
    }

    void update(auto& points, float dt) {
        float tree_span = 0;
        st::internal::unroll_for<Rank>([&](auto i) {
            tree_span =
                std::max<float>(tree_span, this->boundary_.stops[i] - this->boundary_.starts[i]);
        });
        float tree_span_squared = tree_span * tree_span;

        uint64_t n_threads = std::max<int>(std::thread::hardware_concurrency() * 4, 64);
        uint64_t num_points_per_thread = points.size() / n_threads;
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, num_points_per_thread)
        for (auto& point : points) {
            point.acceleration.fill(0);
            const std::function<void(uint64_t, float)> update_recursively = [&](auto branch_index,
                                                                                auto span_squared) {
                const auto update_data = [&](const n_body_tree_data_external& data) {
                    float distance_squared =
                        st::internal::euclidean_distance_squared_arr<CoordinateType, Rank>(
                            point.position, data.position);

                    static constexpr float EpsilonSquared = 1;
                    distance_squared += EpsilonSquared;
                    float reciprocal = 1.0 / std::sqrt(distance_squared);
                    float reciprocal_squared = reciprocal * reciprocal;
                    float coeff = data.mass * reciprocal * reciprocal_squared;
                    st::internal::unroll_for<Rank>([&](auto i) {
                        point.acceleration[i] += (data.position[i] - point.position[i]) * coeff;
                    });
                };

                std::array<bool, spatial_tree_type::BranchingFactor> recurse;
                const auto& branch = this->branches_[branch_index];
                st::internal::unroll_for<spatial_tree_type::BranchingFactor>([&](auto quad) {
                    const auto& branch_ = this->branches_[branch.index_of_first_child + quad];
                    const auto& branch_data__ = branch_data_[branch.index_of_first_child + quad];
                    if (!branch_data__.mass) {
                        recurse[quad] = false;
                        return;
                    }

                    float distance_squared =
                        st::internal::euclidean_distance_squared_arr<CoordinateType, Rank>(
                            point.position, branch_data__.position);
                    if (distance_squared >= span_squared) {
                        update_data(branch_data__);
                        recurse[quad] = false;
                        return;
                    }

                    if (branch_.is_terminal()) [[likely]] {
                        recurse[quad] = false;
                        auto& leaf = leaf_data_[branch_.index()];
                        std::array<CoordinateType, MaximumLeafSize> tmp;
                        tmp.fill(1);
                        for (uint64_t i = 0; i < Rank; ++i) {
                            float       x = point.position[i];
                            float32x4_t x_splat = vdupq_n_f32(x);
                            _Pragma("clang loop unroll_count(2)") for (int64_t j = 0; j < leaf.size;
                                                                       j += 4) {
                                float32x4_t p = vld1q_f32(&leaf.positions[i][j]);
                                float32x4_t delta = vsubq_f32(p, x_splat);
                                float32x4_t delta_squared = vmulq_f32(delta, delta);
                                float32x4_t old = vld1q_f32(&tmp[j]);
                                float32x4_t s = vaddq_f32(delta_squared, old);
                                vst1q_f32(&tmp[j], s);
                            }
                        }
                        _Pragma("clang loop unroll_count(2)") for (int64_t j = 0; j < leaf.size;
                                                                   j += 4) {
                            float32x4_t t = vld1q_f32(&tmp[j]);
                            float32x4_t m = vld1q_f32(&leaf.masses[j]);
                            float32x4_t reciprocal = vrsqrteq_f32(t);
                            float32x4_t reciprocal_squared = vmulq_f32(reciprocal, reciprocal);
                            float32x4_t reciprocal_denom =
                                vmulq_f32(reciprocal_squared, reciprocal);
                            float32x4_t coeff = vmulq_f32(m, reciprocal_denom);
                            vst1q_f32(&tmp[j], coeff);
                        }

                        for (uint64_t i = 0; i < Rank; ++i) {
                            float       x = point.position[i];
                            float32x4_t x_splat = vdupq_n_f32(x);
                            _Pragma("clang loop unroll_count(2)") for (int64_t j = 0; j < leaf.size;
                                                                       j += 4) {
                                float32x4_t t = vld1q_f32(&tmp[j]);
                                float32x4_t p = vld1q_f32(&leaf.positions[i][j]);
                                float32x4_t delta = vsubq_f32(p, x_splat);
                                float32x4_t to_reduce = vmulq_f32(delta, t);
                                float       contribution = reduce_sum(to_reduce);
                                point.acceleration[i] += contribution;
                            }
                        }
                        return;
                    }

                    recurse[quad] = true;
                    return;
                });
                st::internal::unroll_for<spatial_tree_type::BranchingFactor>([&](auto quad) {
                    if (recurse[quad]) [[unlikely]] {
                        update_recursively(branch.index_of_first_child + quad, span_squared / 4);
                    }
                });
            };
            update_recursively(0, tree_span_squared / 4);
            st::internal::unroll_for<Rank>([&](auto i) {
                point.velocity[i] += point.acceleration[i] * dt;
                point.position[i] += point.velocity[i] * dt;
            });
        }
    }

    std::vector<n_body_tree_data_external> branch_data_;
    std::vector<n_body_tree_leaf_simd>     leaf_data_;
};

static std::vector<n_body_data> generate_galaxy(uint64_t                         N,
                                                std::array<CoordinateType, Rank> window_size) {
    CoordinateType span = std::numeric_limits<CoordinateType>::max();
    st::internal::unroll_for<Rank>([&](auto i) { span = std::min(span, window_size[i]); });
    const double                           galaxy_radius = span / 2;
    const double                           arm_tightness = 10.0 / span;
    static constexpr int                   arms = 2;
    static constexpr double                spread = 0.4;
    static constexpr double                max_velocity = 10.0;
    static constexpr double                radius_scale = 5.0;
    std::default_random_engine             gen;
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    std::normal_distribution<double>       normal(0.0, spread);
    std::vector<n_body_data>               points;

    for (uint64_t i = 0; i < N; ++i) {
        int    arm = i % arms;
        double r = galaxy_radius * std::sqrt(uniform(gen));
        double theta = r * arm_tightness + (2 * M_PI / arms) * arm;
        theta += normal(gen);
        r += normal(gen);
        double v_orbit = max_velocity * (1.0 - std::exp(-r / radius_scale));

        n_body_data data;
        data.mass = 1;

        st::internal::unroll_for<Rank>([&](auto i) {
            data.position[i] = window_size[i] / 2;
            if (i == 0) {
                data.position[i] += r * std::cos(theta);
                data.velocity[i] = -v_orbit * std::sin(theta);
            } else if (i == 1) {
                data.position[i] += r * std::sin(theta);
                data.velocity[i] = v_orbit * std::cos(theta);
            } else {
                data.position[i] = 0;
                data.velocity[i] = 0;
            }
        });

        points.push_back(data);
    }

    return points;
}

static std::vector<n_body_data> generate_plummer(uint64_t                         N,
                                                 std::array<CoordinateType, Rank> window_size) {
    CoordinateType span = std::numeric_limits<CoordinateType>::max();
    st::internal::unroll_for<Rank>([&](auto i) { span = std::min(span, window_size[i]); });
    span /= 2;

    std::default_random_engine                     gen;
    std::uniform_real_distribution<CoordinateType> uniform(0.0, 1.0);
    static constexpr CoordinateType                Velocity = 30;
    std::vector<n_body_data>                       points(N);
    for (uint64_t i = 0; i < N; ++i) {
        CoordinateType theta = uniform(gen) * 2 * M_PI;

        n_body_data data;
        data.mass = 10;

        CoordinateType dist = uniform(gen);
        CoordinateType x = std::cos(theta) * dist;
        CoordinateType y = std::sin(theta) * dist;

        data.position[0] = x * span + window_size[0] / 2;
        data.position[1] = y * span + window_size[1] / 2;

        CoordinateType denom = dist * dist + 0.2;
        data.velocity[0] = -y / denom * Velocity;
        data.velocity[1] = x / denom * Velocity;

        points[i] = data;
    }

    return points;
}

// Define a struct for a 3D vector
typedef struct {
    float x, y, z;
} vec3f;

// Function to add two vectors
vec3f vec3f_add(vec3f a, vec3f b) { return (vec3f){a.x + b.x, a.y + b.y, a.z + b.z}; }

// Function to multiply a vector by a scalar
vec3f vec3f_scale(vec3f v, float s) { return (vec3f){v.x * s, v.y * s, v.z * s}; }

// Clamp a value between 0 and 1
float clamp(float v, float min, float max) {
    if (v < min) return min;
    if (v > max) return max;
    return v;
}

// Convert a float in the range [0, 1] to an 8-bit color channel
uint8_t float_to_color_channel(float v) { return (uint8_t)(clamp(v, 0.0f, 1.0f) * 255.0f); }

// The inferno function
vec3f inferno(float t) {
    // Coefficients
    const vec3f c0 = {0.00021894037f, 0.0016510046f, -0.019480899f};
    const vec3f c1 = {0.10651341949f, 0.5639564368f, 3.9327123889f};
    const vec3f c2 = {11.6024930825f, -3.972853966f, -15.94239411f};
    const vec3f c3 = {-41.703996131f, 17.436398882f, 44.354145199f};
    const vec3f c4 = {77.1629356994f, -33.40235894f, -81.80730926f};
    const vec3f c5 = {-71.319428245f, 32.626064264f, 73.209519858f};
    const vec3f c6 = {25.1311262248f, -12.24266895f, -23.07032500f};

    // Polynomial evaluation
    vec3f result = vec3f_add(c0, vec3f_scale(c1, t));
    result = vec3f_add(result, vec3f_scale(c2, t * t));
    result = vec3f_add(result, vec3f_scale(c3, t * t * t));
    result = vec3f_add(result, vec3f_scale(c4, t * t * t * t));
    result = vec3f_add(result, vec3f_scale(c5, t * t * t * t * t));
    result = vec3f_add(result, vec3f_scale(c6, t * t * t * t * t * t));

    // Convert to linear color space
    return result;
}

// Function to convert a value between 0 and 1 into an inferno RGB color
Color inferno_to_rgb(float value) {
    // Clamp value between 0 and 1
    value = clamp(value, 0.0f, 1.0f);

    // Get the inferno color
    vec3f color = inferno(value);

    // Convert to 8-bit RGB channels
    Color c;
    c.r = float_to_color_channel(color.x);
    c.g = float_to_color_channel(color.y);
    c.b = float_to_color_channel(color.z);
    c.a = 255;

    return c;
}

int main(int argc, char** argv) {
    assert(argc == 3);
    const uint32_t number_of_points = std::atoi(argv[1]);
    const bool     draw = std::atoi(argv[2]);

    static constexpr uint64_t        WindowWidth = 1620;
    static constexpr uint64_t        WindowHeight = 1080;
    std::array<CoordinateType, Rank> window_size = {WindowWidth, WindowHeight};
    std::vector<n_body_data>         galaxy = generate_plummer(number_of_points, window_size);

    InitWindow(WindowWidth, WindowHeight, "n-body");

    BeginDrawing();
    ClearBackground(RAYWHITE);
    EndDrawing();

    barnes_hut solver;
    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);

        solver.clear();
        solver.build(galaxy);
        solver.propagate();
        solver.sort_and_vectorize(galaxy);
        solver.update(galaxy, 0.05);

        if (draw) {
            for (auto point : galaxy) {
                auto magnitude =
                    std::pow(st::internal::euclidean_distance_squared_arr<CoordinateType, Rank>(
                                 point.velocity, {0, 0}),
                             0.48);
                DrawPixel(point.position[0], point.position[1], inferno_to_rgb(magnitude / 64));
            }
        }

        DrawFPS(10, 10);
        EndDrawing();
    }

    CloseWindow();

    return 0;
}
