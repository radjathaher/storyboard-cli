#!/usr/bin/env sh
set -eu

repo="radjathaher/storyboard-cli"
bin="storyboard"
install_dir="${INSTALL_DIR:-$HOME/.local/bin}"
version="${VERSION:-latest}"

os="$(uname -s | tr '[:upper:]' '[:lower:]')"
arch="$(uname -m)"

case "$os:$arch" in
  darwin:arm64) target="darwin-aarch64" ;;
  linux:x86_64) target="linux-x86_64" ;;
  *) echo "unsupported platform: $os/$arch" >&2; exit 1 ;;
esac

if [ "$version" = "latest" ]; then
  url="https://github.com/$repo/releases/latest/download/storyboard-cli-$target.tar.gz"
else
  url="https://github.com/$repo/releases/download/$version/storyboard-cli-$version-$target.tar.gz"
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

mkdir -p "$install_dir"
curl -fsSL "$url" | tar -xz -C "$tmp"
install "$tmp/$bin" "$install_dir/$bin"
echo "installed $bin to $install_dir/$bin"
