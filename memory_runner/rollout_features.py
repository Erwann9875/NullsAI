def canonical(rows):
    """Express each proposed tile in the left lane, rotating the lane-specific observations with it."""
    right=rows[:,20]>=.5
    values=rows.copy()
    values[right,20]=1-rows[right,20]
    for l,r in ((10,11),(13,14)):
        values[right,l]=rows[right,r]
        values[right,r]=rows[right,l]
    values[right,26:35]=rows[right,35:44]
    values[right,35:44]=rows[right,26:35]
    return values
