from src.roles.validator import Validator


node1 = Validator([], [], upnp=False, off_chain_test=True, debug=True)
node2 = Validator([], [], upnp=False, off_chain_test=True, debug=True)
node1.start()
node2.start()

print(node1.rsa_key_hash)
print(node2.rsa_key_hash)

node1.connect_node(b"58ef79797cd451e19df4a73fbd9871797f9c6a2995783c7f6fd2406978a2ba2e", node2.host, node2.port)

node1.ping_node(node1.nodes[b"58ef79797cd451e19df4a73fbd9871797f9c6a2995783c7f6fd2406978a2ba2e"])

node1.stop()
node2.stop()
