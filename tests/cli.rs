#![forbid(unsafe_code)]

//! Integration tests for the opt-in `delaunay` companion binary.

#[cfg(feature = "cli")]
mod cli_tests {
    use std::{
        fs,
        path::{Path, PathBuf},
        process::{Command, Output},
        time::{SystemTime, UNIX_EPOCH},
    };

    use serde_json::Value;

    fn delaunay_command() -> Command {
        Command::new(env!("CARGO_BIN_EXE_delaunay"))
    }

    fn run_cli(args: &[&str]) -> Output {
        delaunay_command()
            .args(args)
            .output()
            .expect("delaunay binary should run")
    }

    fn output_text(bytes: &[u8]) -> String {
        String::from_utf8_lossy(bytes).into_owned()
    }

    fn assert_success(output: &Output) {
        assert!(
            output.status.success(),
            "expected success, got status {:?}\nstderr:\n{}",
            output.status.code(),
            output_text(&output.stderr)
        );
    }

    fn assert_exit_code(output: &Output, code: i32) {
        assert_eq!(
            output.status.code(),
            Some(code),
            "unexpected status\nstdout:\n{}\nstderr:\n{}",
            output_text(&output.stdout),
            output_text(&output.stderr)
        );
    }

    fn stdout_json(output: &Output) -> Value {
        serde_json::from_slice(&output.stdout).expect("stdout should contain JSON")
    }

    fn assert_stderr_contains(output: &Output, expected: &str) {
        let stderr = output_text(&output.stderr);
        assert!(
            stderr.contains(expected),
            "stderr should contain {expected:?}, got:\n{stderr}"
        );
    }

    fn assert_generated_triangulation_json(json: &Value, vertex_count: usize) {
        assert_eq!(
            json["vertices"].as_array().map(Vec::len),
            Some(vertex_count)
        );
        assert!(
            json["simplex_vertices"]
                .as_object()
                .is_some_and(|simplex_vertices| !simplex_vertices.is_empty())
        );
    }

    fn target_artifact_path(label: &str, extension: &str) -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after UNIX epoch")
            .as_nanos();
        PathBuf::from("target")
            .join("cli-tests")
            .join(format!("{label}-{stamp}.{extension}"))
    }

    fn target_json_path(label: &str) -> PathBuf {
        target_artifact_path(label, "json")
    }

    fn file_json(path: &Path) -> Value {
        let bytes = fs::read(path).expect("JSON output file should be readable");
        serde_json::from_slice(&bytes).expect("output file should contain JSON")
    }

    #[test]
    fn binary_help_lists_supported_subcommands() {
        let output = run_cli(&["--help"]);
        assert_success(&output);

        let help = output_text(&output.stdout);
        assert!(help.contains("Generate and diagnose d-dimensional Delaunay triangulations"));
        assert!(help.contains("generate"));
        assert!(help.contains("spherical-hero"));
        assert!(help.contains("validation-demo"));
        assert!(help.contains("pachner-stress"));
    }

    #[test]
    fn generate_triangulation_emits_json_to_stdout() {
        let output = run_cli(&[
            "generate",
            "triangulation",
            "--dimension",
            "2",
            "--vertices",
            "4",
            "--seed",
            "1",
        ]);
        assert_success(&output);

        let json = stdout_json(&output);
        assert_generated_triangulation_json(&json, 4);
    }

    #[test]
    fn generate_ball_distribution_emits_points_inside_unit_ball() {
        let output = run_cli(&[
            "generate",
            "triangulation",
            "--dimension",
            "3",
            "--vertices",
            "8",
            "--distribution",
            "ball",
            "--seed",
            "11",
        ]);
        assert_success(&output);

        let json = stdout_json(&output);
        let vertices = json["vertices"]
            .as_array()
            .expect("triangulation JSON should include vertices");
        assert_eq!(vertices.len(), 8);
        for vertex in vertices {
            let point = vertex["point"]
                .as_array()
                .expect("vertex JSON should include coordinates");
            assert_eq!(point.len(), 3);

            let norm_sq: f64 = point
                .iter()
                .map(|coordinate| {
                    let coordinate = coordinate
                        .as_f64()
                        .expect("coordinate should be a JSON number");
                    coordinate * coordinate
                })
                .sum();
            assert!(
                norm_sq <= 1.0 + 1.0e-12,
                "ball distribution emitted point outside unit ball: norm_sq={norm_sq}"
            );
        }
    }

    #[test]
    fn generate_convex_hull_emits_schema_json_to_stdout() {
        let output = run_cli(&[
            "generate",
            "convex-hull",
            "--dimension",
            "3",
            "--vertices",
            "6",
            "--seed",
            "1",
        ]);
        assert_success(&output);

        let json = stdout_json(&output);
        let facets = json["facets"]
            .as_array()
            .expect("convex-hull JSON should include facets");
        assert_eq!(json["schema"], "delaunay.convex_hull");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["dimension"], 3);
        assert_eq!(json["vertex_count"], 6);
        assert_eq!(
            json["facet_count"].as_u64(),
            Some(u64::try_from(facets.len()).expect("facet count should fit in u64"))
        );
        assert!(!facets.is_empty());
    }

    #[test]
    fn generate_visualization_emits_generic_schema_json_to_stdout() {
        let output = run_cli(&[
            "generate",
            "visualization",
            "--dimension",
            "3",
            "--vertices",
            "6",
            "--seed",
            "1",
        ]);
        assert_success(&output);

        let json = stdout_json(&output);
        let metadata = &json["metadata"];
        let simplices = json["simplices"]
            .as_array()
            .expect("visualization JSON should include simplices");
        let adjacency = json["adjacency"]
            .as_array()
            .expect("visualization JSON should include adjacency");

        assert_eq!(metadata["schema"], "delaunay.simplicial_complex");
        assert_eq!(metadata["schema_version"], 1);
        assert_eq!(metadata["dimension"], 3);
        assert_eq!(metadata["vertex_count"], 6);
        assert!(!simplices.is_empty());
        assert_eq!(adjacency.len(), simplices.len() * 4);
        assert!(
            json["vertices"]
                .as_array()
                .is_some_and(|vertices| vertices.iter().all(|vertex| {
                    vertex["id"].is_string()
                        && vertex["coordinates"]
                            .as_array()
                            .is_some_and(|coordinates| coordinates.len() == 3)
                }))
        );
    }

    #[test]
    fn spherical_hero_emits_valid_s2_triangles_to_stdout() {
        let output = run_cli(&["spherical-hero", "--vertices", "8"]);
        assert_success(&output);

        let json = stdout_json(&output);
        let vertices = json["vertices"]
            .as_array()
            .expect("spherical hero JSON should include vertices");
        let simplices = json["simplices"]
            .as_array()
            .expect("spherical hero JSON should include simplices");

        assert_eq!(json["schema"], "delaunay.spherical_hero");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["intrinsic_dimension"], 2);
        assert_eq!(json["ambient_dimension"], 3);
        assert_eq!(vertices.len(), 8);
        let exported_vertex_count =
            u64::try_from(vertices.len()).expect("small fixture count should fit u64");
        assert!(vertices.iter().all(|vertex| {
            let coordinates = vertex
                .as_array()
                .expect("spherical hero vertex should be a coordinate array");
            let squared_norm = coordinates
                .iter()
                .map(|coordinate| {
                    let value = coordinate
                        .as_f64()
                        .expect("spherical hero coordinate should be finite f64 JSON");
                    value * value
                })
                .sum::<f64>();
            coordinates.len() == 3 && (squared_norm - 1.0).abs() <= 1.0e-12
        }));
        assert_eq!(simplices.len(), 2 * vertices.len() - 4);
        assert!(simplices.iter().all(|simplex| {
            simplex.as_array().is_some_and(|vertex_indices| {
                vertex_indices.len() == 3
                    && vertex_indices.iter().all(|index| {
                        index
                            .as_u64()
                            .is_some_and(|index| index < exported_vertex_count)
                    })
                    && vertex_indices[0] != vertex_indices[1]
                    && vertex_indices[0] != vertex_indices[2]
                    && vertex_indices[1] != vertex_indices[2]
            })
        }));

        let repeated = run_cli(&["spherical-hero", "--vertices", "8"]);
        assert_success(&repeated);
        assert_eq!(output.stdout, repeated.stdout);
    }

    #[test]
    fn spherical_hero_accepts_minimum_vertex_count() {
        let output = run_cli(&["spherical-hero", "--vertices", "4"]);
        assert_success(&output);

        let json = stdout_json(&output);
        assert_eq!(json["vertices"].as_array().map(Vec::len), Some(4));
        assert_eq!(json["simplices"].as_array().map(Vec::len), Some(4));
    }

    #[test]
    fn spherical_hero_writes_requested_json_artifact() {
        let path = target_json_path("spherical-hero");
        let output = run_cli(&[
            "spherical-hero",
            "--vertices",
            "8",
            "--output",
            path.to_str().expect("target path should be UTF-8"),
        ]);
        assert_success(&output);
        assert!(
            output.stdout.is_empty(),
            "--output should keep spherical hero JSON out of stdout"
        );

        let json = file_json(&path);
        assert_eq!(json["schema"], "delaunay.spherical_hero");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["intrinsic_dimension"], 2);
        assert_eq!(json["ambient_dimension"], 3);
        assert_eq!(json["vertices"].as_array().map(Vec::len), Some(8));
    }

    #[test]
    fn spherical_hero_rejects_too_few_vertices() {
        for vertices in ["0", "3"] {
            let output = run_cli(&["spherical-hero", "--vertices", vertices]);
            assert_exit_code(&output, 1);
            assert_stderr_contains(
                &output,
                &format!("2D generation requires at least 4 vertices, got {vertices}"),
            );
        }
    }

    #[test]
    fn spherical_hero_rejects_empty_output_path_during_parsing() {
        let output = run_cli(&["spherical-hero", "--vertices", "8", "--output", ""]);
        assert_exit_code(&output, 2);
        assert_stderr_contains(&output, "a value is required");
        assert_stderr_contains(&output, "--output <OUTPUT>");
    }

    #[test]
    fn generate_triangulation_writes_requested_json_artifact() {
        let path = target_json_path("generate-triangulation");
        let output = run_cli(&[
            "generate",
            "triangulation",
            "--dimension",
            "2",
            "--vertices",
            "4",
            "--seed",
            "1",
            "--output",
            path.to_str().expect("target path should be UTF-8"),
        ]);
        assert_success(&output);
        assert!(
            output.stdout.is_empty(),
            "--output should keep generated triangulation JSON out of stdout"
        );

        let json = file_json(&path);
        assert_generated_triangulation_json(&json, 4);
    }

    #[test]
    fn validation_demo_writes_requested_json_artifact() {
        let path = target_json_path("validation-demo");
        let output = run_cli(&[
            "validation-demo",
            "--output",
            path.to_str().expect("target path should be UTF-8"),
        ]);
        assert_success(&output);
        assert!(
            output.stdout.is_empty(),
            "--output should keep validation-demo JSON out of stdout"
        );

        let json = file_json(&path);
        let cases = json["cases"]
            .as_array()
            .expect("validation-demo JSON should include cases");
        assert_eq!(json["schema"], "delaunay.validation_demo");
        assert_eq!(json["schema_version"], 1);
        assert_eq!(json["dimension"], 2);
        assert_eq!(json["valid_baseline"]["status"], "passed");
        assert_eq!(cases.len(), 5);
        assert_eq!(cases[3]["layer"], "Valid realization");
    }

    #[test]
    fn pachner_stress_accepts_small_quiet_summary_run() {
        let summary_path = target_json_path("pachner-stress-summary");
        let progress_path = target_artifact_path("pachner-stress-progress", "csv");
        let output = run_cli(&[
            "pachner-stress",
            "--dimension",
            "3d",
            "--mode",
            "round-trip",
            "--vertices",
            "5",
            "--attempts",
            "1",
            "--validate-every",
            "1",
            "--key-refresh-every",
            "1",
            "--retry-attempts",
            "4",
            "--seed",
            "7",
            "--quiet",
            "--progress-csv",
            progress_path.to_str().expect("target path should be UTF-8"),
            "--summary-json",
            summary_path.to_str().expect("target path should be UTF-8"),
        ]);
        assert_success(&output);
        assert!(
            output.stdout.is_empty(),
            "--quiet should suppress Pachner stress telemetry"
        );

        let json = file_json(&summary_path);
        assert_eq!(json["dimension"], 3);
        assert_eq!(json["label"], "3d");
        assert_eq!(json["mode"], "round-trip");
        assert_eq!(json["validation_scope"], "topology");
        assert_eq!(json["configured_vertices"], 5);
        assert_eq!(json["attempts"], 1);
        assert_eq!(json["validate_every"], 1);
        assert_eq!(json["key_refresh_every"], 1);
        assert_eq!(json["retry_attempts"], 4);
        assert_eq!(json["seed"], 7);
        assert_eq!(json["source"]["mode"], "round-trip");
        assert_eq!(json["source"]["validation_scope"], "topology");
        assert_eq!(json["report"]["attempts"], 1);

        let progress = fs::read_to_string(progress_path).expect("progress CSV should be readable");
        let rows: Vec<_> = progress.lines().collect();
        assert_eq!(
            rows.len(),
            2,
            "expected one CSV header and one progress row"
        );
        assert_eq!(
            rows[0],
            "dimension,label,mode,validation_scope,sequence,step,attempts,accepted,rejected,candidate_misses,\
             proposal_rejections,validations,validation_nanos,acceptance_rate,vertices,simplices"
        );
        let fields: Vec<_> = rows[1].split(',').collect();
        assert_eq!(fields.len(), 16);
        assert_eq!(fields[0], "3");
        assert_eq!(fields[1], "3d");
        assert_eq!(fields[2], "round-trip");
        assert_eq!(fields[3], "topology");
        assert_eq!(fields[4], "1");
        assert_eq!(fields[5], "1");
        assert_eq!(fields[6], "1");
        assert_eq!(fields[11], "1");
    }

    #[test]
    fn pachner_stress_accepts_small_random_walk_summary_run() {
        let summary_path = target_json_path("pachner-stress-random-walk-summary");
        let output = run_cli(&[
            "pachner-stress",
            "--dimension",
            "3d",
            "--mode",
            "random-walk",
            "--vertices",
            "5",
            "--attempts",
            "2",
            "--validate-every",
            "1",
            "--key-refresh-every",
            "1",
            "--retry-attempts",
            "4",
            "--seed",
            "7",
            "--quiet",
            "--summary-json",
            summary_path.to_str().expect("target path should be UTF-8"),
        ]);
        assert_success(&output);
        assert!(
            output.stdout.is_empty(),
            "--quiet should suppress Pachner stress telemetry"
        );

        let json = file_json(&summary_path);
        assert_eq!(json["dimension"], 3);
        assert_eq!(json["label"], "3d");
        assert_eq!(json["mode"], "random-walk");
        assert_eq!(json["validation_scope"], "topology");
        assert_eq!(json["configured_vertices"], 5);
        assert_eq!(json["attempts"], 2);
        assert_eq!(json["validate_every"], 1);
        assert_eq!(json["source"]["mode"], "random-walk");
        assert_eq!(json["source"]["validation_scope"], "topology");
        assert_eq!(json["report"]["attempts"], 2);
        assert_eq!(json["report"]["validations"], 2);
    }

    #[test]
    fn pachner_stress_emits_setup_stage_telemetry() {
        let output = run_cli(&[
            "pachner-stress",
            "--dimension",
            "3d",
            "--mode",
            "round-trip",
            "--vertices",
            "5",
            "--attempts",
            "1",
            "--validate-every",
            "1",
            "--key-refresh-every",
            "1",
            "--retry-attempts",
            "4",
            "--seed",
            "7",
        ]);
        assert_success(&output);

        let stdout = output_text(&output.stdout);
        assert!(stdout.contains("pachner_stress_stage"));
        assert!(stdout.contains("validation_scope=topology"));
        assert!(stdout.contains("stage=generate_points_start"));
        assert!(stdout.contains("stage=construction_start"));
        assert!(stdout.contains("stage=initial_topology_validation_done"));
        assert!(stdout.contains("pachner_stress_source"));
        assert!(stdout.contains("pachner_stress_progress"));
        assert!(stdout.contains("pachner_stress_metric"));
    }

    #[test]
    fn generate_rejects_unsupported_dimension_after_parsing() {
        let output = run_cli(&["generate", "--dimension", "6", "--vertices", "8"]);
        assert_exit_code(&output, 1);
        assert_stderr_contains(&output, "generate supports dimensions 2 through 5, got 6");
    }

    #[test]
    fn generate_rejects_too_few_vertices_for_dimension() {
        let output = run_cli(&["generate", "--dimension", "3", "--vertices", "3"]);
        assert_exit_code(&output, 1);
        assert_stderr_contains(&output, "3D generation requires at least 4 vertices, got 3");
    }

    #[test]
    fn generate_rejects_zero_vertices() {
        let output = run_cli(&["generate", "--dimension", "3", "--vertices", "0"]);
        assert_exit_code(&output, 1);
        assert_stderr_contains(&output, "3D generation requires at least 4 vertices, got 0");
    }

    #[test]
    fn generate_rejects_empty_output_path_during_parsing() {
        let output = run_cli(&[
            "generate",
            "triangulation",
            "--dimension",
            "3",
            "--vertices",
            "4",
            "--output",
            "",
        ]);
        assert_exit_code(&output, 2);
        assert_stderr_contains(&output, "a value is required");
        assert_stderr_contains(&output, "--output <OUTPUT>");
    }

    #[test]
    fn generate_rejects_invalid_distribution_value() {
        let output = run_cli(&["generate", "--dimension", "3", "--distribution", "sphere"]);
        assert_exit_code(&output, 2);
        assert_stderr_contains(&output, "invalid value 'sphere'");
        assert_stderr_contains(&output, "[possible values: cube, ball]");
    }

    #[test]
    fn pachner_stress_rejects_unsupported_dimension_value() {
        let output = run_cli(&["pachner-stress", "--dimension", "2d"]);
        assert_exit_code(&output, 2);
        assert_stderr_contains(&output, "invalid value '2d'");
        assert_stderr_contains(&output, "[possible values: 3d, 4d]");
    }

    #[test]
    fn pachner_stress_rejects_zero_validated_arguments() {
        for (argument, expected) in [
            ("--attempts", "--attempts must be positive"),
            ("--validate-every", "--validate-every must be positive"),
            (
                "--key-refresh-every",
                "--key-refresh-every must be positive",
            ),
            ("--retry-attempts", "--retry-attempts must be positive"),
        ] {
            let output = run_cli(&["pachner-stress", argument, "0", "--quiet"]);
            assert_exit_code(&output, 1);

            let stderr = output_text(&output.stderr);
            assert!(
                stderr.contains(expected),
                "{argument} stderr should contain {expected:?}, got:\n{stderr}"
            );
        }
    }

    #[test]
    fn pachner_stress_rejects_too_few_vertices_for_dimension() {
        let output = run_cli(&["pachner-stress", "--dimension", "3d", "--vertices", "3"]);
        assert_exit_code(&output, 1);
        assert_stderr_contains(&output, "3D stress requires at least 4 vertices, got 3");
    }

    #[test]
    fn pachner_stress_rejects_zero_vertices() {
        let output = run_cli(&["pachner-stress", "--dimension", "3d", "--vertices", "0"]);
        assert_exit_code(&output, 1);
        assert_stderr_contains(&output, "3D stress requires at least 4 vertices, got 0");
    }

    #[test]
    fn pachner_stress_rejects_duplicate_artifact_paths() {
        let path = target_json_path("pachner-stress-duplicate-artifact");
        let path = path.to_str().expect("target path should be UTF-8");
        let output = run_cli(&[
            "pachner-stress",
            "--quiet",
            "--progress-csv",
            path,
            "--summary-json",
            path,
        ]);

        assert_exit_code(&output, 1);
        assert_stderr_contains(
            &output,
            "progress CSV and summary JSON must use different paths",
        );
    }

    #[test]
    fn unsupported_topology_style_flags_are_rejected() {
        for (args, expected) in [
            (
                &["generate", "--spherical"][..],
                "unexpected argument '--spherical'",
            ),
            (
                &["validation-demo", "--hyperbolic"][..],
                "unexpected argument '--hyperbolic'",
            ),
        ] {
            let output = run_cli(args);
            assert_exit_code(&output, 2);
            assert_stderr_contains(&output, expected);
        }
    }
}
