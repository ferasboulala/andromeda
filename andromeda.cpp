#include <cstdlib>
#include <iostream>
#include <random>
#include <thread>
#include <type_traits>

#include "raylib.h"
#include "simde/arm/neon.h"
#include "spatial-tree/spatial-tree.h"

static constexpr int Rank = 2;
static constexpr int MaximumLeafSize = 64;
using CoordinateType = float;
using MassType = float;

static_assert(std::is_floating_point_v<CoordinateType>);

struct particle {
    std::array<CoordinateType, Rank> acceleration;
    std::array<CoordinateType, Rank> velocity;
    std::array<CoordinateType, Rank> position;
    MassType                         mass;
};

struct barnes_hut_branch_data {
    MassType                         mass;
    std::array<CoordinateType, Rank> position;
};

struct barnes_hut_leaf_data {
    MassType mass;
    uint32_t idx;
};

struct barnes_hut_leaf_simd_data {
    uint32_t                                                      size;
    std::array<std::array<CoordinateType, MaximumLeafSize>, Rank> positions;
    std::array<MassType, MaximumLeafSize>                         masses;
};

class solver {
public:
    virtual ~solver() = default;
    virtual void step(std::vector<particle>& particles, double dt) = 0;
};

using spatial_tree_type = st::internal::
    spatial_tree<CoordinateType, barnes_hut_leaf_data, Rank, MaximumLeafSize, 32, true>;
class barnes_hut_solver_impl : public spatial_tree_type {
public:
    barnes_hut_solver_impl(bool skip_leaves) : skip_leaves_(skip_leaves) {}
    ~barnes_hut_solver_impl() = default;

    void step(auto& particles, double dt) {
        build(particles);
        propagate();
        sort_and_vectorize(particles);
        update(particles, dt);
    }

    void build(const auto& points) {
        clear();

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
            this->emplace(point.position, barnes_hut_leaf_data{point.mass, i++});
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
                              [&](const auto& data) { branch_data_entry.mass += data.mass; });
                if (branch_data_entry.mass == 0.0) {
                    continue;
                }

                const MassType reciprocal = 1.0 / branch_data_entry.mass;
                for (uint64_t i = 0; i < leaf.size; ++i) {
                    const MassType weight = leaf.items[i].mass * reciprocal;
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
        std::vector<particle> sorted(points.size());
        /// TODO: Parallelize this
        for (uint32_t i = 0, j = 0; i < this->leaves_.size(); ++i) {
            auto& leaf = this->leaves_[i];
            auto& leaf_data = leaf_data_[i];
            leaf_data.size = leaf.size;

            for (uint32_t k = 0; k < leaf.size; ++k, ++j) {
                sorted[j] = points[leaf.items[k].idx];
                leaf.items[k].idx = j;
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

    static inline float reduce_sum(simde_float32x4_t vec) {
        float32x2_t sum_pair = simde_vadd_f32(vget_low_f32(vec), vget_high_f32(vec));
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
        uint64_t num_leaves_per_thread = leaves_.size() / n_threads;
#pragma omp parallel for num_threads(n_threads) schedule(dynamic, num_leaves_per_thread)
        for (const auto& leaf : leaves_) {
            if (leaf.size == 0) continue;
            st::bounding_box<CoordinateType, Rank> box = {
                std::numeric_limits<CoordinateType>::max(),
                std::numeric_limits<CoordinateType>::max(),
                std::numeric_limits<CoordinateType>::lowest(),
                std::numeric_limits<CoordinateType>::lowest()};
            for (uint64_t i = 0; i < leaf.size; ++i) {
                auto& point = points[leaf.items[i].idx];
                point.acceleration.fill(0);
                st::internal::unroll_for<Rank>([&](auto j) {
                    box.starts[j] = std::min(box.starts[j], point.position[j]);
                    box.stops[j] = std::max(box.stops[j], point.position[j]);
                });
            }

            const std::function<void(uint64_t, float)> update_recursively = [&](auto branch_index,
                                                                                auto span_squared) {
                const auto update_data = [&](const barnes_hut_branch_data& data) {
                    /// TODO: SIMD this.
                    for (uint64_t j = 0; j < leaf.size; ++j) {
                        auto& point = points[leaf.items[j].idx];
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
                    }
                };
                std::array<bool, spatial_tree_type::BranchingFactor> recurse;
                recurse.fill(false);
                const auto& branch = this->branches_[branch_index];
                st::internal::unroll_for<spatial_tree_type::BranchingFactor>([&](auto quad) {
                    const auto& branch_ = this->branches_[branch.index_of_first_child + quad];
                    const auto& branch_data__ = branch_data_[branch.index_of_first_child + quad];
                    if (!branch_data__.mass) {
                        return;
                    }

                    auto distance_squared = box.sdistance(branch_data__.position);
                    if (distance_squared >= span_squared) {
                        update_data(branch_data__);
                        return;
                    }

                    if (branch_.is_terminal()) {
                        if (skip_leaves_) {
                            return;
                        }

                        for (uint64_t foo = 0; foo < leaf.size; ++foo) {
                            auto& point = points[leaf.items[foo].idx];
                            auto& leaf = leaf_data_[branch_.index()];
                            std::array<CoordinateType, MaximumLeafSize> tmp;
                            tmp.fill(1);
                            for (uint64_t i = 0; i < Rank; ++i) {
                                float             x = point.position[i];
                                simde_float32x4_t x_splat = simde_vdupq_n_f32(x);
                                _Pragma("clang loop unroll_count(2)") for (int64_t j = 0;
                                                                           j < leaf.size;
                                                                           j += 4) {
                                    simde_float32x4_t p = simde_vld1q_f32(&leaf.positions[i][j]);
                                    simde_float32x4_t delta = simde_vsubq_f32(p, x_splat);
                                    simde_float32x4_t delta_squared = simde_vmulq_f32(delta, delta);
                                    simde_float32x4_t old = simde_vld1q_f32(&tmp[j]);
                                    simde_float32x4_t s = simde_vaddq_f32(delta_squared, old);
                                    vst1q_f32(&tmp[j], s);
                                }
                            }
                            _Pragma("clang loop unroll_count(2)") for (int64_t j = 0; j < leaf.size;
                                                                       j += 4) {
                                simde_float32x4_t t = simde_vld1q_f32(&tmp[j]);
                                simde_float32x4_t m = simde_vld1q_f32(&leaf.masses[j]);
                                simde_float32x4_t reciprocal = simde_vrsqrteq_f32(t);
                                simde_float32x4_t reciprocal_squared =
                                    vmulq_f32(reciprocal, reciprocal);
                                simde_float32x4_t reciprocal_denom =
                                    vmulq_f32(reciprocal_squared, reciprocal);
                                simde_float32x4_t coeff = simde_vmulq_f32(m, reciprocal_denom);
                                vst1q_f32(&tmp[j], coeff);
                            }

                            for (uint64_t i = 0; i < Rank; ++i) {
                                float             x = point.position[i];
                                simde_float32x4_t x_splat = simde_vdupq_n_f32(x);
                                _Pragma("clang loop unroll_count(2)") for (int64_t j = 0;
                                                                           j < leaf.size;
                                                                           j += 4) {
                                    simde_float32x4_t t = simde_vld1q_f32(&tmp[j]);
                                    simde_float32x4_t p = simde_vld1q_f32(&leaf.positions[i][j]);
                                    simde_float32x4_t delta = simde_vsubq_f32(p, x_splat);
                                    simde_float32x4_t to_reduce = simde_vmulq_f32(delta, t);
                                    float             contribution = reduce_sum(to_reduce);
                                    point.acceleration[i] += contribution;
                                }
                            }
                        }
                        return;
                    }

                    recurse[quad] = true;
                });
                st::internal::unroll_for<spatial_tree_type::BranchingFactor>([&](auto quad) {
                    if (recurse[quad]) [[unlikely]] {
                        update_recursively(branch.index_of_first_child + quad, span_squared / 4);
                    }
                });
            };
            update_recursively(0, tree_span_squared / 4);
            for (uint64_t j = 0; j < leaf.size; ++j) {
                auto& point = points[leaf.items[j].idx];
                st::internal::unroll_for<Rank>([&](auto i) {
                    point.velocity[i] += point.acceleration[i] * dt;
                    point.position[i] += point.velocity[i] * dt;
                });
            }
        }
    }

    std::vector<barnes_hut_branch_data>    branch_data_;
    std::vector<barnes_hut_leaf_simd_data> leaf_data_;
    bool                                   skip_leaves_;
};

class barnes_hut_solver : public solver {
public:
    barnes_hut_solver(bool skip_leaves) : impl_(skip_leaves) {}
    ~barnes_hut_solver() = default;

    void step(std::vector<particle>& particles, double dt) override { impl_.step(particles, dt); }

private:
    barnes_hut_solver_impl impl_;
};

static std::vector<particle> generate_galaxy(uint64_t                         N,
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
    std::vector<particle>                  points;

    for (uint64_t i = 0; i < N; ++i) {
        int    arm = i % arms;
        double r = galaxy_radius * std::sqrt(uniform(gen));
        double theta = r * arm_tightness + (2 * M_PI / arms) * arm;
        theta += normal(gen);
        r += normal(gen);
        double v_orbit = max_velocity * (1.0 - std::exp(-r / radius_scale));

        particle data;
        data.mass = 1;
        data.position[0] = window_size[0] / 2;
        data.position[1] = window_size[1] / 2;
        data.position[0] += r * std::cos(theta);
        data.velocity[0] = -v_orbit * std::sin(theta);
        data.position[1] += r * std::sin(theta);
        data.velocity[1] = v_orbit * std::cos(theta);

        points.push_back(data);
    }

    return points;
}

static std::vector<particle> generate_plummer(uint64_t                         N,
                                              std::array<CoordinateType, Rank> window_size) {
    CoordinateType span = std::numeric_limits<CoordinateType>::max();
    st::internal::unroll_for<Rank>([&](auto i) { span = std::min(span, window_size[i]); });
    span /= 2;

    std::default_random_engine                     gen;
    std::uniform_real_distribution<CoordinateType> uniform(0.0, 1.0);
    static constexpr CoordinateType                Velocity = 30;
    std::vector<particle>                          points(N);
    for (uint64_t i = 0; i < N; ++i) {
        CoordinateType theta = uniform(gen) * 2 * M_PI;

        particle data;
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

typedef struct {
    float x, y, z;
} vec3f;

inline vec3f vec3f_add(vec3f a, vec3f b) { return (vec3f){a.x + b.x, a.y + b.y, a.z + b.z}; }

inline vec3f vec3f_scale(vec3f v, float s) { return (vec3f){v.x * s, v.y * s, v.z * s}; }

inline uint8_t float_to_color_channel(float v) {
    return (uint8_t)(std::clamp(v, 0.0f, 1.0f) * 255.0f);
}

inline vec3f inferno(float t) {
    const vec3f c0 = {0.00021894037f, 0.0016510046f, -0.019480899f};
    const vec3f c1 = {0.10651341949f, 0.5639564368f, 3.9327123889f};
    const vec3f c2 = {11.6024930825f, -3.972853966f, -15.94239411f};
    const vec3f c3 = {-41.703996131f, 17.436398882f, 44.354145199f};
    const vec3f c4 = {77.1629356994f, -33.40235894f, -81.80730926f};
    const vec3f c5 = {-71.319428245f, 32.626064264f, 73.209519858f};
    const vec3f c6 = {25.1311262248f, -12.24266895f, -23.07032500f};

    vec3f result = vec3f_add(c0, vec3f_scale(c1, t));
    result = vec3f_add(result, vec3f_scale(c2, t * t));
    result = vec3f_add(result, vec3f_scale(c3, t * t * t));
    result = vec3f_add(result, vec3f_scale(c4, t * t * t * t));
    result = vec3f_add(result, vec3f_scale(c5, t * t * t * t * t));
    result = vec3f_add(result, vec3f_scale(c6, t * t * t * t * t * t));

    return result;
}

Color inferno_to_rgb(float value) {
    value = std::clamp(value, 0.0f, 1.0f);
    vec3f color = inferno(value);

    Color c;
    c.r = float_to_color_channel(color.x);
    c.g = float_to_color_channel(color.y);
    c.b = float_to_color_channel(color.z);
    c.a = 255;

    return c;
}

#define ENV_SOLVER                      "ANDROMEDA_SOLVER"
#define ENV_SOLVER_NAIVE                "NAIVE"
#define ENV_SOLVER_BARNES_HUT           "BARNESHUT"
#define ENV_SOLVER_BARNES_HUT_NO_LEAVES "BARNESHUT2"
#define ENV_SOLVER_DEFAULT              ENV_SOLVER_BARNES_HUT
#define ENV_TARGET                      "ANDROMEDA_TARGET"
#define ENV_TARGET_CPU                  "CPU"
#define ENV_TARGET_GPU                  "GPU"
#define ENV_TARGET_DEFAULT              ENV_TARGET_CPU
#define ENV_INIT_SHAPE                  "ANDROMEDA_INIT_SHAPE"
#define ENV_INIT_SHAPE_GALAXY           "GALAXY"
#define ENV_INIT_SHAPE_PLUMMER          "PLUMMER"
#define ENV_INIT_SHAPE_DEFAULT          ENV_INIT_SHAPE_GALAXY

static void print_usage(std::string_view program_name) {
    std::cerr << "Usage: " << program_name << " <command> <number_of_particles> [-d]\n";
    std::cerr << "  <command>            : one of {run, bench}\n";
    std::cerr << "  <number_of_particles>: an integer value > 0\n";
    std::cerr << "  -d                   : optional debug flag\n";
}

enum class Solver { Naive, BarnesHut, BarnesHutNoLeaves };

enum class Target { CPU, GPU };

enum class InitShape { Galaxy, Plummer };

struct cli_arguments {
    bool      run;
    uint64_t  number_of_particles;
    bool      debug;
    Solver    solver;
    Target    target;
    InitShape shape;
};

static inline std::string getenv_or_default(const char* name, std::string_view default_value) {
    auto ret = std::getenv(name);
    if (!ret) {
        return std::string(default_value);
    }

    return ret;
}

static std::optional<cli_arguments> parse_cli_arguments(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        print_usage(argv[0]);
        return std::nullopt;
    }

    cli_arguments          ret;
    const std::string_view command(argv[1]);
    if (command == "run") {
        ret.run = true;
    } else if (command == "bench") {
        ret.run = false;
    } else {
        print_usage(argv[0]);
        return std::nullopt;
    }

    ret.number_of_particles = std::stoull(argv[2]);
    if (ret.number_of_particles == 0) {
        print_usage(argv[0]);
        return std::nullopt;
    }

    if (argc == 3) {
        ret.debug = false;
    } else {
        const std::string_view debug(argv[3]);
        if (debug == "-d") {
            ret.debug = true;
        } else {
            print_usage(argv[0]);
            return std::nullopt;
        }
    }

    auto target = getenv_or_default(ENV_TARGET, ENV_TARGET_DEFAULT);
    auto solver = getenv_or_default(ENV_SOLVER, ENV_SOLVER_DEFAULT);
    auto init_shape = getenv_or_default(ENV_INIT_SHAPE, ENV_INIT_SHAPE_DEFAULT);

    if (target == ENV_TARGET_CPU) {
        ret.target = Target::CPU;
    } else if (target == ENV_TARGET_GPU) {
        ret.target = Target::GPU;
    } else {
        print_usage(argv[0]);
        return std::nullopt;
    }

    if (solver == ENV_SOLVER_NAIVE) {
        ret.solver = Solver::Naive;
    } else if (solver == ENV_SOLVER_BARNES_HUT) {
        ret.solver = Solver::BarnesHut;
    } else if (solver == ENV_SOLVER_BARNES_HUT_NO_LEAVES) {
        ret.solver = Solver::BarnesHutNoLeaves;
    } else {
        print_usage(argv[0]);
        return std::nullopt;
    }

    if (init_shape == ENV_INIT_SHAPE_GALAXY) {
        ret.shape = InitShape::Galaxy;
    } else if (init_shape == ENV_INIT_SHAPE_PLUMMER) {
        ret.shape = InitShape::Plummer;
    } else {
        print_usage(argv[0]);
        return std::nullopt;
    }

    return ret;
}

static void run(const cli_arguments& args) {
    static constexpr uint64_t DefaultWindowWidth = 1024;
    static constexpr uint64_t DefaultWindowHeight = 1024;

    SetConfigFlags(FLAG_WINDOW_RESIZABLE);
    InitWindow(DefaultWindowWidth, DefaultWindowHeight, "window");

    BeginDrawing();
    ClearBackground(RAYWHITE);
    EndDrawing();

    auto generator = args.shape == InitShape::Galaxy ? generate_galaxy : generate_plummer;
    auto particles =
        generator(args.number_of_particles,
                  std::array<CoordinateType, Rank>{DefaultWindowWidth, DefaultWindowHeight});

    std::unique_ptr<solver> s;
    if (args.solver == Solver::Naive) {
        std::cerr << "Currently not implemented\n";
        return;
    } else {
        s = std::make_unique<barnes_hut_solver>(args.solver == Solver::BarnesHutNoLeaves);
    }

    while (!WindowShouldClose()) {
        BeginDrawing();
        ClearBackground(BLACK);

        s->step(particles, 0.05);
        for (auto particle : particles) {
            auto magnitude =
                std::pow(st::internal::euclidean_distance_squared_arr<CoordinateType, Rank>(
                             particle.velocity, {0, 0}),
                         0.48);
            DrawPixel(particle.position[0], particle.position[1], inferno_to_rgb(magnitude / 64));
        }

        DrawFPS(10, 10);
        EndDrawing();
    }

    CloseWindow();
}

static void bench(const cli_arguments& args) {}

int main(int argc, char** argv) {
    auto args = parse_cli_arguments(argc, argv);
    if (!args) {
        return 1;
    }

    if (args->run) {
        run(*args);
    } else {
        bench(*args);
    }

    return 0;
}
