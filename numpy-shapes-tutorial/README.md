Insights:

Passing in scalar shape and single-element tuple is the same

    In [60]: np.zeros(6)
    Out[60]: array([0., 0., 0., 0., 0., 0.])

    In [61]: np.zeros((6,))
    Out[61]: array([0., 0., 0., 0., 0., 0.])

Indexing is in the same order as axes in shape

    In [63]: pp=np.zeros((4,5,6))

    # Last element
    In [64]: pp[3,4,5]
    Out[64]: 0.0
