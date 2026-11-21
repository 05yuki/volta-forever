#!/bin/sh
# volta-forever: install a release for the current user.
#
#   curl -fsSL https://raw.githubusercontent.com/nbald/volta-forever/main/install.sh | bash
#
# It downloads the archive of the version below from the project's releases, checks its
# SHA-256 against the release's SHA256SUMS, unpacks it under $VF_PREFIX, links the commands
# `vf` and `volta-forever` in $VF_BIN, runs `vf fetch` (which downloads NVIDIA's cuBLAS for
# CUDA 12.9.2 from NVIDIA's servers, 1 GB, under NVIDIA's licence) and ends with `vf doctor`.
# It installs no package of the system and asks for no password: what is missing is said.
#
# Settings, in the environment:
#   VF_VERSION        the version to install (default: the one this script was published with)
#   VF_PREFIX         where versions are unpacked (default: ~/.local/share/volta-forever;
#                     /opt/volta-forever for root)
#   VF_BIN            where the commands are linked (default: ~/.local/bin; /usr/local/bin
#                     for root)
#   VF_INSTALL_FETCH  0: do not run `vf fetch`
#   VF_INSTALL_BASE   where the archive and SHA256SUMS are taken from (default: the release
#                     of that version on GitHub)
#
# To remove it: delete $VF_PREFIX and the two links in $VF_BIN.
set -eu

say() { printf 'volta-forever: %s\n' "$*"; }
fail() { printf 'volta-forever: %s\n' "$*" >&2; exit 1; }

# (Everything is in a function that is called on the last line: a script cut short while it
# was being downloaded runs nothing.)
main() {
    version=${VF_VERSION:-0.1.0}
    name=volta-forever-$version-linux-x86_64
    archive=$name.tar.gz
    base=${VF_INSTALL_BASE:-https://github.com/nbald/volta-forever/releases/download/v$version}

    # --- what this needs of the machine ---
    [ "$(uname -s)" = Linux ] || fail "this is for Linux ($(uname -s) here)"
    [ "$(uname -m)" = x86_64 ] || fail "this is for x86-64 ($(uname -m) here)"
    libc=$(getconf GNU_LIBC_VERSION 2>/dev/null | sed -n 's/^glibc //p')
    [ -n "$libc" ] || fail "the C library of this system is not glibc: the binaries need glibc 2.34 or later"
    libc_major=${libc%%.*}
    libc_minor=${libc#*.}
    libc_minor=${libc_minor%%.*}
    if [ "$libc_major" -lt 2 ] || { [ "$libc_major" -eq 2 ] && [ "$libc_minor" -lt 34 ]; }; then
        fail "glibc $libc is too old: the binaries need 2.34 or later (Ubuntu 22.04, Debian 12, RHEL 9)"
    fi
    for tool in tar gzip sha256sum mktemp; do
        command -v "$tool" >/dev/null 2>&1 || fail "$tool is needed and was not found"
    done
    if command -v curl >/dev/null 2>&1; then
        download() { curl -fsSL --retry 3 -o "$2" "$1"; }
    elif command -v wget >/dev/null 2>&1; then
        download() { wget -q -O "$2" "$1"; }
    else
        fail "curl or wget is needed and neither was found"
    fi

    if [ "$(id -u)" = 0 ]; then
        prefix=${VF_PREFIX:-/opt/volta-forever}
        bin=${VF_BIN:-/usr/local/bin}
    else
        prefix=${VF_PREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/volta-forever}
        bin=${VF_BIN:-$HOME/.local/bin}
    fi

    # --- download and check ---
    work=$(mktemp -d "${TMPDIR:-/tmp}/volta-forever-install.XXXXXX") || fail "no temporary directory"
    trap 'rm -rf "$work"' EXIT
    say "downloading $archive from $base"
    download "$base/$archive" "$work/$archive" || fail "could not download $base/$archive"
    download "$base/SHA256SUMS" "$work/SHA256SUMS" || fail "could not download $base/SHA256SUMS"
    expected=$(sed -n "s/^\([0-9a-f]\{64\}\)[ *]*$archive\$/\1/p" "$work/SHA256SUMS" | head -1)
    [ -n "$expected" ] || fail "SHA256SUMS of the release does not name $archive"
    got=$(sha256sum "$work/$archive" | cut -d' ' -f1)
    [ "$got" = "$expected" ] || fail "the SHA-256 of $archive is $got, and the release says $expected: nothing was installed"

    # --- unpack, beside what is there, then put it in place ---
    tar -xzf "$work/$archive" -C "$work" || fail "could not unpack $archive"
    [ -x "$work/$name/bin/vf" ] || fail "$archive does not hold $name/bin/vf"
    mkdir -p "$prefix" "$bin" || fail "cannot write in $prefix or $bin (VF_PREFIX and VF_BIN name other places)"
    rm -rf "$prefix/$name.new"
    mv "$work/$name" "$prefix/$name.new" || fail "cannot write in $prefix"
    # NVIDIA's libraries that an installation already downloaded are kept: this version's own
    # first, else those of another version.
    for old in "$prefix/$name" "$prefix"/volta-forever-*-linux-x86_64; do
        if [ -d "$old/cuda-12.9.2" ] && [ ! -e "$prefix/$name.new/cuda-12.9.2" ] && [ "$old" != "$prefix/$name.new" ]; then
            mv "$old/cuda-12.9.2" "$prefix/$name.new/cuda-12.9.2"
            say "kept the CUDA 12.9.2 libraries of $old"
        fi
    done
    rm -rf "$prefix/$name"
    mv "$prefix/$name.new" "$prefix/$name"
    ln -sfn "$prefix/$name/bin/vf" "$bin/vf"
    ln -sfn "$prefix/$name/bin/volta-forever" "$bin/volta-forever"
    say "installed $(cat "$prefix/$name/VERSION" 2>/dev/null || echo "$version") in $prefix/$name"
    say "commands: $bin/vf, $bin/volta-forever"

    # --- NVIDIA's cuBLAS for CUDA 12.9.2 ---
    if [ "${VF_INSTALL_FETCH:-1}" != 0 ]; then
        if [ -f "$prefix/$name/cuda-12.9.2/lib/libcublas.so.12" ]; then
            say "NVIDIA's cuBLAS 12.9.2 is there already"
        elif ! command -v python3 >/dev/null 2>&1; then
            say "python3 was not found: install it, then run: $bin/vf fetch"
        elif ! command -v xz >/dev/null 2>&1; then
            say "xz was not found (package xz-utils, or xz): install it, then run: $bin/vf fetch"
        else
            say "downloading NVIDIA's cuBLAS for CUDA 12.9.2 from NVIDIA's servers (1 GB), under NVIDIA's licence"
            if "$bin/vf" fetch; then
                say "NVIDIA's licence is in $prefix/$name/cuda-12.9.2/LICENSE"
            else
                say "vf fetch did not end well: run it again with: $bin/vf fetch"
            fi
        fi
    fi

    # --- what is there ---
    echo
    "$bin/vf" doctor || true
    echo
    case ":$PATH:" in
        *":$bin:"*) ;;
        *) say "$bin is not on your PATH: add it with   export PATH=\"$bin:\$PATH\"" ;;
    esac
    say "run a program with   vf python3 your_program.py"
    say "a Python environment that runs through the layer by itself:   vf venv .venv"
}

main "$@"
