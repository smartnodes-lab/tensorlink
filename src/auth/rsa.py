from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend


def generate_rsa_key_pair():
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048
    )

    # Save private key to a file
    with open("private_key.pem", "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))

    with open("public_key.pem", "wb") as f:
        f.write(private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))


def load_public_key(path):
    with open(path, "rb") as f:
        return serialization.load_pem_public_key(
            f.read(),
            backend=default_backend()
        )


def authenticate_public_key(public_key) -> bool:
    try:
        public_key = serialization.load_pem_public_key(
            public_key.encode(),
            backend=default_backend()
        )

        if public_key.public_numbers().e != 65537:
            return False

        if not isinstance(public_key, rsa.RSAPublicKey):
            return False

        return True

    except Exception:
        return False


