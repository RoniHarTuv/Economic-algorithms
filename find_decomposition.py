# RONI HAR TUV
# Assignment11_Ex3

import cvxpy

"""
    This function decide whether a budget vector can be decomposed into equal-share
    contributions that respect each citizen’s preference list.

    We need a non-negative matrix d[i,j] satisfying the follow :
        (1)  ∑_i d[i,j]  =  budget[j] for every organisation j
        (2)  ∑_j d[i,j]  =  C ⁄ n same total share for every citizen i
        (3)  d[i,j] > 0  only if j ∈ preferences[i]

    Parameters:
    budget : list[float]
        Required amount for each organization (Σ budget = C).
    preferences : list[set[int]]
        preferences[i] is the set of organizations citizen i want.

    Returns:
    bool,
        True  ⇔  such a matrix d exists; False otherwise.
    """


def find_decomposition(budget: list[float], preferences: list[set[int]]) -> bool:
    m = len(budget)  # number of organizations
    n = len(preferences)  # number of citizens
    if m == 0 or n == 0:
        return False

    total = sum(budget)  # C
    share = total / n  # C / n

    # Decision variables: d[i,j] = contribution of citizen i to organization j
    d = cvxpy.Variable((n, m))

    constraints = []
    # (1) each organization receives exactly its budget
    for j in range(m):
        constraints.append(cvxpy.sum(d[:, j]) == budget[j])
    # (2) every citizen pays exactly the same share
    for i in range(n):
        constraints.append(cvxpy.sum(d[i, :]) == share)
    # (3) citizens gives only to preferred organizations
    for i in range(n):
        for j in range(m):
            if j in preferences[i]:
                constraints.append(d[i, j] >= 0)
            else:
                constraints.append(d[i, j] == 0)

    prob = cvxpy.Problem(cvxpy.Minimize(0), constraints)
    prob.solve()

    return prob.status in (cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE)


if __name__ == "__main__":
    # Example that is decomposable - from the assignment
    budget = [400, 50, 50, 0]  #C = 500
    preferences = [
        {0, 1},  # citizen 0
        {0, 2},  # citizen 1
        {0, 3},  # citizen 2
        {1, 2},  # citizen 3
        {0}  # citizen 4
    ]
    print("Test-1 (should be True):",
          find_decomposition(budget, preferences))

    # Example that is not decomposable
    bad_preferences = [
        {0},  # citizen 0
        {1},  # citizen 1
        {2},  # citizen 2
        {3},  # citizen 3
        {0}  # citizen 4
    ]
    print("Test-2 (should be False):",
          find_decomposition(budget, bad_preferences))
