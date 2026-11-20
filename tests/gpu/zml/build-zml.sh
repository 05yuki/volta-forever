#!/usr/bin/env bash
# Build ZML's LLM example as upstream has it (CUDA 13 plugin, sandbox of CUDA libraries made by the
# build itself), optionally with patches, for tests/gpu/zml.sh. Needs Docker and no GPU.
#
#   tests/gpu/zml/build-zml.sh TAG [PATCH...]
#   tests/gpu/zml/build-zml.sh fix1004 tests/gpu/zml/zml-8898051c7b-close-open-outputs.diff
#
# (That patch is what lets the example run a Qwen3.5 model at this commit: qwen35-open-sharding.md.)
#
# Result: $VF_ZML_OUT/<commit>-TAG/ (default $VF_TEST_ROOT/zml/; give that directory to
# tests/gpu/zml.sh as $VF_ZML): `llm`, `llm.runfiles/` with every link resolved, and `PICKS`
# saying what was built. The first build downloads and compiles for an hour; the Bazel files stay
# in $VF_ZML_BAZEL (default /dev/shm/zml-bazel-stock, tens of GB) and make the next ones a
# matter of minutes.
set -euo pipefail

W=$(cd "$(dirname "$0")/../../.." && pwd)
. "$W/tests/gpu/env.sh"
tag=${1:?usage: build-zml.sh TAG [PATCH...]}
shift
out=${VF_ZML_OUT:-$VF_TEST_ROOT/zml}
root=${VF_ZML_BAZEL:-/dev/shm/zml-bazel-stock}

ZML_REPO=https://github.com/zml/zml.git
ZML_COMMIT=8898051c7b6b8cd015cdce8e75afc5404396405f   # main, 2026-11-04
BAZELISK_VERSION=v1.25.0
BAZELISK_SHA256=fd8fdff418a1758887520fa42da7e6ae39aefc788cf5e7f7bb8db6934d279fc4
BUILD_IMAGE=${VF_ZML_IMAGE:-volta-forever-zml-build:ubuntu24.04}
build_flags="--config=release --@zml//platforms:cuda=true --jobs=40 --curses=no --color=no --show_progress_rate_limit=60"

src=/dev/shm/zml-src-$tag
dest=$out/${ZML_COMMIT:0:10}-$tag

[ -d "$src/.git" ] || git clone --quiet "$ZML_REPO" "$src"
git -C "$src" fetch --quiet origin 2>/dev/null || true
git -C "$src" checkout --quiet --force --detach "$ZML_COMMIT"
git -C "$src" clean --quiet -fdx
patches=()
for patch in "$@"; do
    patch=$(readlink -f "$patch")
    git -C "$src" apply "$patch"
    patches+=("$(basename "$patch") sha256 $(sha256sum "$patch" | cut -c1-64)")
done
echo "sources: $(git -C "$src" rev-parse --short HEAD), ${#patches[@]} patch(es): $(git -C "$src" status --short | tr '\n' ' ')"

bazelisk=${XDG_CACHE_HOME:-$HOME/.cache}/volta-forever/bazelisk/bazelisk-$BAZELISK_VERSION
if [ ! -x "$bazelisk" ]; then
    mkdir -p "$(dirname "$bazelisk")"
    curl -fsSL -o "$bazelisk.part" "https://github.com/bazelbuild/bazelisk/releases/download/$BAZELISK_VERSION/bazelisk-linux-amd64"
    echo "$BAZELISK_SHA256  $bazelisk.part" | sha256sum -c --strict --quiet
    chmod +x "$bazelisk.part" && mv "$bazelisk.part" "$bazelisk"
fi
docker inspect --type=image "$BUILD_IMAGE" >/dev/null 2>&1 || docker build --quiet --tag "$BUILD_IMAGE" - >/dev/null <<'DOCKERFILE'
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates python3 curl unzip xz-utils zstd \
 && rm -rf /var/lib/apt/lists/*
DOCKERFILE

mkdir -p "$root"/{out,tmp,home,repo-cache,bazelisk}
install=$root/install-$tag
if [ -e "$install" ]; then chmod -R u+w "$install" && rm -rf "${install:?}"; fi
docker run --rm --user "$(id -u):$(id -g)" --memory 90g \
    -v "$src":/build/src -v "$root":/build/root -v "$root/tmp":/tmp -v "$root/tmp":/var/tmp \
    -v "$bazelisk":/usr/local/bin/bazel:ro \
    -e HOME=/build/root/home -e BAZELISK_HOME=/build/root/bazelisk \
    -w /build/src "$BUILD_IMAGE" bash -euo pipefail -c "
bazel --output_user_root=/build/root/out build --repository_cache=/build/root/repo-cache $build_flags //examples/llm
mkdir /build/root/install-$tag
cp -L bazel-bin/examples/llm/llm /build/root/install-$tag/llm
cp -rL bazel-bin/examples/llm/llm.runfiles /build/root/install-$tag/llm.runfiles
"

if [ -e "$dest" ]; then chmod -R u+w "$dest" && rm -rf "${dest:?}"; fi
mkdir -p "$dest"
rsync -a --chmod=u+w,a+rX "$install/" "$dest/"
chmod -R u+w "$install" && rm -rf "${install:?}"
{
    echo "$ZML_REPO $ZML_COMMIT"
    if [ ${#patches[@]} -eq 0 ]; then echo "no patch"; else printf 'patch %s\n' "${patches[@]}"; fi
    echo "bazel build $build_flags //examples/llm"
    echo "installed with cp -L (links of the Bazel output resolved); the sandbox of CUDA libraries is the build's own"
} >"$dest/PICKS"
echo "built $dest ($(du -sh "$dest" | cut -f1))"
