#!/usr/bin/env python3


if __name__ == "__main__":
    # add a warning messege that this functionality is under development and should not be used in production.
    # make it visually distinct, e.g. with a different color or formatting.
    import warnings

    from TB2J.magnon.magnon3 import main

    warnings.warn(
        """ 
        # !!!!!!!!!!!!!!!!!! WARNING: =============================
        # 
        # This functionality is under development and should not be used in production.
        # It is provided for testing and development purposes only.
        # Please use with caution and report any issues to the developers.
        #
        # This warning will be removed in future releases.
        # =====================================

        """,
        UserWarning,
        stacklevel=2,
    )
    # Call the main function from the magnons module
    main()
