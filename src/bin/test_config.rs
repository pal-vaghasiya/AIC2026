use config::Environment;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct Config {
    pub upstream: UpstreamConfig,
}

#[derive(Debug, Deserialize)]
pub struct UpstreamConfig {
    pub default_target_url: String,
    pub api_key: String,
}

fn main() {
    std::env::set_var("CP_UPSTREAM__DEFAULT_TARGET_URL", "https://generativelanguage.googleapis.com/v1beta/openai/");
    std::env::set_var("CP_UPSTREAM__API_KEY", "AQ.Ab8RN6LFPvMxFKQKWwplObGaovODRH_wDNlYCEddmn5nt4CX5Q");

    let builder = config::Config::builder()
        .set_default("upstream.default_target_url", "https://generativelanguage.googleapis.com/v1beta/openai/").unwrap()
        .set_default("upstream.api_key", "").unwrap()
        .add_source(config::Environment::with_prefix("CP").separator("__"));

    let cfg: Config = builder.build().unwrap().try_deserialize().unwrap();
    println!("{:#?}", cfg);
}
