from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
import os


def generate_rsa_key_pair() -> None:
    path = "keys"

    if not os.path.exists(os.path.join(path, "public_key.pem")):
        # Save private and public rsa keys to files
        with open(os.path.join(path, "private_key.pem"), "wb") as f:
            key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048,
                backend=default_backend()
            )

            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ))

        with open(os.path.join(path, "public_key.pem"), "wb") as f:
            f.write(key.public_key().public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ))


def load_public_key():
    path = "keys/public_key.pem"
    with open(path, "rb") as f:
        return serialization.load_pem_public_key(
            f.read(),
            backend=default_backend()
        )


def load_private_key():
    path = "keys/private_key.pem"
    with open(path, "rb") as f:
        return serialization.load_pem_private_key(
            f.read(),
            backend=default_backend(),
            password=None
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

    except Exception as e:
        return False


def get_public_key_bytes(public_key) -> bytes:
    public_key_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return public_key_bytes.decode()


def get_private_key_bytes(private_key) -> bytes:
    private_key_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL
    )
    return private_key_bytes.decode()


def get_public_key_obj(public_key_bytes):
    return serialization.load_pem_public_key(public_key_bytes, backend=default_backend())


def get_private_key_obj(private_key_bytes):
    return serialization.load_pem_private_key(private_key_bytes, backend=default_backend(), password=None)
