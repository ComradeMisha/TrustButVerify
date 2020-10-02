"""Tests for alpacka.runner."""


from alpacka import runner


def test_smoke(tmpdir, capsys):
    n_epochs = 3
    runner.Runner(
        output_dir=tmpdir,
        n_envs=2,
        n_epochs=n_epochs,
        log_graph_distances=False,
        log_n_experienced_states=False,
    ).run()

    # Check that metrics were printed in each epoch.
    captured = capsys.readouterr()
    assert captured.out.count('return_mean') == n_epochs
