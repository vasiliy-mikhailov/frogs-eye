# Build the static-ish binary, then ship it on a bare runtime image.
FROM rust:1-slim-bookworm AS build
WORKDIR /src
COPY Cargo.toml ./
COPY src ./src
RUN cargo build --release && strip target/release/meterproxy || true

FROM debian:bookworm-slim
RUN useradd -r -u 10001 meter
COPY --from=build /src/target/release/meterproxy /usr/local/bin/meterproxy
USER meter
EXPOSE 8091
ENTRYPOINT ["/usr/local/bin/meterproxy"]
