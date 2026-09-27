import numpy as np

from llm_society.simulation.population import generate_population


def test_n_agents_created(make_cfg):
    cfg = make_cfg(population_sizes=[7, 23])
    for n in (7, 23):
        pop = generate_population(cfg, n, 0)
        assert len(pop.profiles) == n
        assert len({p.agent_id for p in pop.profiles}) == n
        assert all(set(p.stance) == set(cfg.dimension_names) for p in pop.profiles)


def test_stratified_covers_space_and_is_not_perfectly_correlated(make_cfg):
    cfg = make_cfg(population_sizes=[100])
    pop = generate_population(cfg, 100, 0)
    z = np.array([p.latent_position for p in pop.profiles])
    # one agent per stratum of [-1, 1]
    counts, _ = np.histogram(z, bins=np.linspace(-1, 1, 101))
    assert (counts == 1).all()
    X = np.array([[p.stance[d] for d in cfg.dimension_names] for p in pop.profiles])
    corr = np.corrcoef(X.T)
    off = corr[~np.eye(len(corr), dtype=bool)]
    assert off.max() < 0.999
    assert X.min() < 20 and X.max() > 80


def test_ids_are_shuffled_relative_to_position(make_cfg):
    pop = generate_population(make_cfg(population_sizes=[50]), 50, 0)
    z = [p.latent_position for p in pop.profiles]
    assert z != sorted(z)


def test_same_seed_same_population(make_cfg):
    cfg = make_cfg()
    a, b = generate_population(cfg, 8, 0), generate_population(cfg, 8, 0)
    assert a.to_dict() == b.to_dict()
    c = generate_population(cfg, 8, 1)
    assert c.to_dict()["profiles"] != a.to_dict()["profiles"]


def test_all_initialisation_methods(make_cfg, tmp_path):
    import pandas as pd

    for method in ["uniform", "normal", "bimodal", "clustered"]:
        pop = generate_population(make_cfg(initialisation={"method": method}), 12, 0)
        assert len(pop.profiles) == 12
    f = tmp_path / "custom.csv"
    cfg = make_cfg()
    pd.DataFrame({d: np.linspace(0, 100, 20) for d in cfg.dimension_names}).to_csv(f, index=False)
    pop = generate_population(make_cfg(initialisation={"method": "custom_file", "custom_file": str(f)}), 12, 0)
    assert len(pop.profiles) == 12
