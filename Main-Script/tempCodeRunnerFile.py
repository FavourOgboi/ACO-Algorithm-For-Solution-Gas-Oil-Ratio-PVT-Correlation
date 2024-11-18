

def gor_objective(params, pressures, observed_gor):
    """
    Calculates error between estimated and observed GOR using given parameters.
    :param params: Tuple of constants for the GOR formula.
    :param pressures: Array of pressure data points.