if __name__ == "__main__":

    validator = Validator(
        debug=True,
        upnp=False,
        off_chain_test=True,
        private_key="1c6768059a3e77d68a2dc3a075c93161803dbe2ad3b72069b6801a1db3a8a8f8",
    )

    # validator2 = Validator(
    #     ip,
    #     port + 1,
    #     debug=True,
    #     upnp=False,
    #     off_chain_test=False,
    #     private_key="d23f58a739fe6dd1390191a67e11d3d681e048168d985c9acb5609bff1f799ea",
    # )

    validator.start()
