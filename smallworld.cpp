#include "power.hpp"
#include "union_find.hpp"
#include <alps/parapack/parapack.h>
#include <alps/parapack/worker.h>

using cluster::power2;
using cluster::power4;

class bond {
public:
  bond(unsigned int N, unsigned int source, unsigned int target) {
    source_ = source % N;
    target_ = target % N;
if (source_ > target_) std::swap(source_, target_);
  }
  bool operator<(bond const& rhs) const {
    return (source_ < rhs.source_) || ((source_ == rhs.source_) && (target_ < rhs.target_));
  }
  unsigned int source() const { return source_; }
  unsigned int target() const { return target_; }
  bool operator==(bond const& rhs) const {
    return (source_ == rhs.source_) && (target_ == rhs.target_);
  }
private:
  unsigned int source_, target_;
};

class cluster_worker : public alps::parapack::mc_worker {
public:
  typedef alps::parapack::mc_worker super_t;
  typedef cluster::union_find::node cluster_fragment_t;
  cluster_worker(alps::Parameters const& p) : super_t(p), mcs(p) {
    // random number generator for random samples
    boost::mt19937 eng(boost::lexical_cast<long>(p["DISORDER_SEED"]));
    boost::variate_generator<boost::mt19937&, boost::uniform_real<> >
      disorder(eng, boost::uniform_real<>());

    // lattice structure
    int lattice_type = boost::lexical_cast<unsigned int>(p["LATTICE_TYPE"]);
    num_sites = boost::lexical_cast<unsigned int>(p["L"]);
    // type 0 bonds
    bonds0.clear();
    num_bonds0 = num_sites;
    for (int i = 0; i < num_bonds0; ++i) {
      bonds0.push_back(bond(num_sites, i, i + 1));
    }
    // type 1 bonds
    bonds1.clear();
    if (lattice_type == 0) {
      // non-random
      num_bonds1 = num_sites / 2;
      for (int i = 0; i < num_bonds1; ++i) {
        bonds1.push_back(bond(num_sites, i, (i + num_sites / 2)));
      }
    } else {
      if (lattice_type == 1) {
        num_bonds1 = num_sites;
      } else {
        num_bonds1 = std::sqrt(1.0 * num_sites);
      }
      std::set<bond> inserted;
      while (true) {
        int source = num_sites * disorder();
        int target = num_sites * disorder();
        bond b(num_sites, source, target);
        if (inserted.find(b) == inserted.end()) {
          bonds1.push_back(b);
          inserted.insert(b);
        }
        if (bonds1.size() == num_bonds1) break;
      }
    }
    
    t = boost::lexical_cast<double>(p["T"]);
    J0 = boost::lexical_cast<double>(p["J0"]);
    J1 = boost::lexical_cast<double>(p["J1"]);
    prob0 = 1 - std::exp(-2 * J0 / t);
    prob1 = 1 - std::exp(-2 * J1 / t);

    spins.assign(num_sites, 1);
    fragments.resize(num_sites);
    config.resize(num_sites);
  }
  virtual ~cluster_worker() {}

  void init_observables(alps::Parameters const&, alps::ObservableSet& obs) {
    obs << alps::RealObservable("Energy Density")
        << alps::RealObservable("Magnetization Density (unimproved)")
        << alps::RealObservable("Magnetization Density^2 (unimproved)")
        << alps::RealObservable("Magnetization Density^4 (unimproved)")
        << alps::RealObservable("Magnetization Density^2")
        << alps::RealObservable("Magnetization Density^4");
  }

  bool is_thermalized() const { return mcs.is_thermalized(); }
  double progress() const { return mcs.progress(); }

  void run(alps::ObservableSet& obs) {
    ++mcs;
    
    std::fill(fragments.begin(), fragments.end(), cluster_fragment_t());
    for (int i = 0; i < num_bonds0; ++i) {
      int source  = bonds0[i].source();
      int target  = bonds0[i].target();
      if (spins[source] == spins[target] && uniform_01() < prob0) unify(fragments, source, target);
    }
    for (int i = 0; i < num_bonds1; ++i) {
      int source  = bonds1[i].source();
      int target  = bonds1[i].target();
      if (spins[source] == spins[target] && uniform_01() < prob1) unify(fragments, source, target);
    }

    // cluster flip
    int nc = set_id(fragments, 0, fragments.size(), 0);
    copy_id(fragments, 0, fragments.size());
    for (int c = 0; c < nc; ++c) config[c] = (uniform_01() < 0.5) ? -1 : 1;
    for (int s = 0; s < num_sites; ++s) spins[s] = config[fragments[s].id()];

    // accumulate physical property of clusters
    double ene = 0;
    for (int i = 0; i < num_bonds0; ++i)
      ene -= J0 * spins[bonds0[i].source()] * spins[bonds0[i].target()];
    for (int i = 0; i < num_bonds1; ++i)
      ene -= J1 * spins[bonds1[i].source()] * spins[bonds1[i].target()];
    ene /= num_sites;

    double m = 0;
    for (int i = 0; i < num_sites; ++i) m += spins[i];
    m /= num_sites;
    
    double m2 = 0;
    double m4 = 0;
    for (int i = 0; i < num_sites; ++i) {
      if (fragments[i].is_root()) {
        double w = fragments[i].weight();
        w /= num_sites;
        m2 += power2(w);
        m4 += power4(w);
      }
    }

    obs["Energy Density"] << ene;
    obs["Magnetization Density (unimproved)"] << m;
    obs["Magnetization Density^2 (unimproved)"] << power2(m);
    obs["Magnetization Density^4 (unimproved)"] << power4(m);
    obs["Magnetization Density^2"] << m2;
    obs["Magnetization Density^4"] << (3 * power2(m2) - 2 * m4);
  }

  void save(alps::ODump& dp) const { dp << mcs << spins; }
  void load(alps::IDump& dp) { dp >> mcs >> spins; }

private:
  // lattice
  int lattice_type;
  int num_sites, num_bonds0, num_bonds1;
  std::vector<bond> bonds0, bonds1;

  // parameters
  double t, J0, J1, prob0, prob1;

  // configuration (checkpoint)
  alps::mc_steps mcs;
  std::vector<int> spins;

  // working vectors
  std::vector<cluster_fragment_t> fragments;
  std::vector<int> config;
};

class cluster_evaluator : public alps::parapack::simple_evaluator {
public:
  cluster_evaluator(alps::Parameters const&) {}
  void evaluate(alps::ObservableSet& obs) const {
    {
      alps::RealObsevaluator m2 = obs["Magnetization Density^2 (unimproved)"];
      alps::RealObsevaluator m4 = obs["Magnetization Density^4 (unimproved)"];
      alps::RealObsevaluator binder("Binder Ratio of Magnetization (unimproved)");
      binder = power2(m2) / m4;
      obs.addObservable(binder);
    }
    {
      alps::RealObsevaluator m2 = obs["Magnetization Density^2"];
      alps::RealObsevaluator m4 = obs["Magnetization Density^4"];
      alps::RealObsevaluator binder("Binder Ratio of Magnetization");
      binder = power2(m2) / m4;
      obs.addObservable(binder);
    }
  }
};

PARAPACK_SET_VERSION("Ising model in a small world");
PARAPACK_SET_COPYRIGHT("  Copyright (C) 2014 by Synge Todo <wistaria@comp-phys.org>");
PARAPACK_REGISTER_WORKER(cluster_worker, "cluster");
PARAPACK_REGISTER_EVALUATOR(cluster_evaluator, "cluster");

int main(int argc, char** argv) { return alps::parapack::start(argc, argv); }
