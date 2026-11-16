// Browser-side LAN gate. The exhibit is opened from a private address; public
// hostnames must not enter live mode unless the operator opts in with ?lan=1.
// Keep in sync with broker/annie_broker/lan.py (RFC1918 + IPv6 ULA).
export const PRIVATE_HOST =
  /^(?:localhost|127(?:\.\d{1,3}){3}|10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}|\[?::1\]?|.+\.local)$/i;

export function isPrivateHost(hostname) {
  return PRIVATE_HOST.test(hostname || "");
}

export function isLocalStage(hostname, params) {
  return isPrivateHost(hostname) || params.get("lan") === "1";
}
