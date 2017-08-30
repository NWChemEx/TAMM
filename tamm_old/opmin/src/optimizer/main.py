import optimizer


def main(trans_unit, use_cse, use_fact, use_refine):
    optimizer.optimize(trans_unit, use_cse, use_fact, use_refine)

    return trans_unit
