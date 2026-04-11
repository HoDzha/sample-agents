import ctypes
import os
from ctypes import wintypes

from openai import OpenAI

DEFAULT_CREDENTIAL_TARGET = "pac1-py-openai"
_CRED_TYPE_GENERIC = 1


class _CredentialAttribute(ctypes.Structure):
    _fields_ = [
        ("Keyword", wintypes.LPWSTR),
        ("Flags", wintypes.DWORD),
        ("ValueSize", wintypes.DWORD),
        ("Value", ctypes.POINTER(ctypes.c_ubyte)),
    ]


class _Credential(ctypes.Structure):
    _fields_ = [
        ("Flags", wintypes.DWORD),
        ("Type", wintypes.DWORD),
        ("TargetName", wintypes.LPWSTR),
        ("Comment", wintypes.LPWSTR),
        ("LastWritten", ctypes.c_byte * 8),
        ("CredentialBlobSize", wintypes.DWORD),
        ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
        ("Persist", wintypes.DWORD),
        ("AttributeCount", wintypes.DWORD),
        ("Attributes", ctypes.POINTER(_CredentialAttribute)),
        ("TargetAlias", wintypes.LPWSTR),
        ("UserName", wintypes.LPWSTR),
    ]


def _read_windows_credential(target_name: str) -> str | None:
    if os.name != "nt":
        return None

    advapi32 = ctypes.WinDLL("Advapi32.dll")
    cred_ptr = ctypes.POINTER(_Credential)()
    advapi32.CredReadW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.POINTER(_Credential)),
    ]
    advapi32.CredReadW.restype = wintypes.BOOL
    advapi32.CredFree.argtypes = [ctypes.c_void_p]
    advapi32.CredFree.restype = None

    ok = advapi32.CredReadW(
        target_name,
        _CRED_TYPE_GENERIC,
        0,
        ctypes.byref(cred_ptr),
    )
    if not ok:
        return None

    try:
        credential = cred_ptr.contents
        size = int(credential.CredentialBlobSize)
        if size <= 0 or not credential.CredentialBlob:
            return None

        raw = ctypes.string_at(credential.CredentialBlob, size)
        return raw.decode("utf-16-le").rstrip("\x00")
    finally:
        advapi32.CredFree(cred_ptr)


def get_openai_credential_target() -> str:
    return os.getenv("OPENAI_CREDENTIAL_TARGET") or DEFAULT_CREDENTIAL_TARGET


def resolve_openai_token() -> str:
    target_name = get_openai_credential_target()
    token = _read_windows_credential(target_name)
    if token:
        return token

    token = os.getenv("OPENAI_ACCESS_TOKEN") or os.getenv("OPENAI_API_KEY")
    if token:
        return token

    raise RuntimeError(
        "Missing OpenAI credentials. Store the secret in Windows Credential "
        f"Manager under '{target_name}' or set OPENAI_ACCESS_TOKEN / "
        "OPENAI_API_KEY as a fallback."
    )


def create_openai_client() -> OpenAI:
    return OpenAI(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=resolve_openai_token(),
    )
