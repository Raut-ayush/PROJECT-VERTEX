def test_package_imports():
    import vertex
    import vertex.simulation

    assert vertex.__doc__
    assert vertex.simulation.PlanarState()
