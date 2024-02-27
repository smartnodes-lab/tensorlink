from substrateinterface import SubstrateInterface
from substrateinterface.contracts import ContractCode, ContractInstance, ContractMetadata
from substrateinterface.exceptions import SubstrateRequestException
from src.cryptography.substrate import load_substrate_keypair
from src.cryptography.rsa import get_rsa_pub_key


chain = SubstrateInterface(url="wss://ws.test.azero.dev")
contract = ContractInstance.create_from_address(
    substrate=chain,
    contract_address="5GMGXFH8W6xe89wmNbtxUNtWhJ7iAaPvMyLGPeta72ue8uaj",
    metadata_file="../src/smartnodes.json"  # Temp path to contract metadata
)
keypair = load_substrate_keypair(
    public_key="test",
    password="test"
)
pub_key = get_rsa_pub_key(True)


try:
    user_exists = contract.exec(keypair, "get_user", {"pub_key": "123"})
    if user_exists:
        print("User exists")
except Exception as e:
    print(str(e))
