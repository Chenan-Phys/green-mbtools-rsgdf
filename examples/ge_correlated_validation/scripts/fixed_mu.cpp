// Controlled diagnostic using unmodified GREEN libraries and solver kernels.
// The standard executable has no fixed-mu CLI parameter. This harness only
// initializes its public Dyson mu() setter before the existing SC loop.
#include <green/mbpt/dyson.h>
#include <green/mbpt/mbpt_run.h>
#include <green/sc/sc_loop.h>
#include <mpi.h>
#include <iomanip>

class fixed_mu_dyson : public green::mbpt::shared_mem_dyson {
public:
  explicit fixed_mu_dyson(const green::params::params& p)
      : green::mbpt::shared_mem_dyson(p) {
    if (p["const_density"].as<bool>())
      throw std::runtime_error("Fixed-mu diagnostic requires const_density=false");
    mu() = p["benchmark_mu"].as<double>();
  }
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  green::utils::context();
  auto p = green::params::params("GREEN fixed-mu backend diagnostic", "upstream-main-2346c87");
  green::sc::define_parameters(p);
  green::symmetry::define_parameters(p);
  green::grids::define_parameters(p);
  green::mbpt::define_parameters(p);
  p.define<double>("benchmark_mu", "Common chemical potential for the controlled comparison", 0.0);
  if (!p.parse(argc, argv)) {
    if (!green::utils::context().global_rank) p.help_or_version();
    MPI_Finalize();
    return 0;
  }
  if (!green::utils::context().global_rank) {
    p.print();
    std::cout << "Controlled chemical potential = " << std::setprecision(17)
              << p["benchmark_mu"].as<double>() << std::endl;
  }
  try {
    green::mbpt::check_input(p);
    green::sc::sc_loop<fixed_mu_dyson> sc(MPI_COMM_WORLD, p);
    fixed_mu_dyson dyson(p);
    using green::mbpt::ztensor;
    auto g = green::utils::shared_object<ztensor<5>>(
        dyson.ft().sd().repn_fermi().nts(), dyson.ns(), dyson.bz_utils().ink(), dyson.nso(), dyson.nso());
    auto st = green::utils::shared_object<ztensor<5>>(
        dyson.ft().sd().repn_fermi().nts(), dyson.ns(), dyson.bz_utils().ink(), dyson.nso(), dyson.nso());
    auto s1 = ztensor<4>(dyson.ns(), dyson.bz_utils().ink(), dyson.nso(), dyson.nso());
    green::mbpt::read_hartree_fock_selfenergy(p, dyson.bz_utils(), s1);
    g.fence();
    if (!green::utils::context().node_rank) g.object().set_zero();
    g.fence();
    st.fence();
    if (!green::utils::context().node_rank) st.object().set_zero();
    st.fence();
    green::mbpt::hf_solver hf(p, dyson.bz_utils(), dyson.S_k());
    const auto type = p["scf_type"].as<green::mbpt::scf_type>();
    if (type == green::mbpt::GW) {
      green::mbpt::gw_solver gw(p, dyson.ft(), dyson.bz_utils(), dyson.S_k());
      green::sc::composition_solver cs(hf, gw);
      sc.solve(cs, dyson.H_k(), dyson.S_k(), g, s1, st);
    } else if (type == green::mbpt::GF2) {
      green::mbpt::gf2_solver gf2(p, dyson.ft(), dyson.bz_utils());
      green::sc::composition_solver cs(hf, gf2);
      sc.solve(cs, dyson.H_k(), dyson.S_k(), g, s1, st);
    } else {
      throw std::runtime_error("Diagnostic supports GF2 and GW only");
    }
    if (!green::utils::context().global_rank) std::cout << "Controlled GREEN calculation completed." << std::endl;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    MPI_Abort(green::utils::context().global, 1);
  }
  MPI_Finalize();
  return 0;
}
