default:
    just --list

prepare:
    zigup master

build: 
    zig build

run:
    zig build run

test: 
    zig build test

clean:
    rm -r ./.zig-cache ./zig-out
