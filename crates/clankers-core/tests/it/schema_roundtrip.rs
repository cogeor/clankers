//! Integration tests for serde roundtrip and `validate_against` on the four
//! schema types.
//!
//! Loop 01 / WS1 PR1 — see `docs/plans/WS1-plan.md` § 6.

use clankers_core::schema::{
    ActionSchema, ActionSemantics, FrameEncoding, FrameSchema, ObservationSchema, ObservationSlot,
    RecorderSchema, SchemaDtype, SchemaMismatch,
};

fn make_observation_schema() -> ObservationSchema {
    ObservationSchema {
        slots: vec![
            ObservationSlot {
                name: "joint_state".into(),
                dtype: SchemaDtype::F32,
                shape: vec![6],
                units: Some("rad".into()),
                source_sensor: "robot_joints".into(),
            },
            ObservationSlot {
                name: "camera".into(),
                dtype: SchemaDtype::U8,
                shape: vec![64, 64, 3],
                units: None,
                source_sensor: "front_camera".into(),
            },
        ],
        version: ObservationSchema::SCHEMA_VERSION,
    }
}

fn make_action_schema(semantics: ActionSemantics, dim: usize) -> ActionSchema {
    ActionSchema {
        semantics,
        dim,
        low: Some(vec![-1.0; dim]),
        high: Some(vec![1.0; dim]),
        version: ActionSchema::SCHEMA_VERSION,
    }
}

fn make_recorder_schema() -> RecorderSchema {
    RecorderSchema {
        channels: vec![
            FrameSchema {
                channel: "/joints".into(),
                message_type: "JointState".into(),
                encoding: FrameEncoding::Json,
                version: FrameSchema::SCHEMA_VERSION,
            },
            FrameSchema {
                channel: "/camera/front".into(),
                message_type: "Image".into(),
                encoding: FrameEncoding::RawBytes,
                version: FrameSchema::SCHEMA_VERSION,
            },
            FrameSchema {
                channel: "/camera/wrist".into(),
                message_type: "Image".into(),
                encoding: FrameEncoding::ProtobufFqn("foo.bar.Image".into()),
                version: FrameSchema::SCHEMA_VERSION,
            },
        ],
        version: RecorderSchema::SCHEMA_VERSION,
    }
}

#[test]
fn observation_schema_serde_roundtrip() {
    let schema = make_observation_schema();
    let json = serde_json::to_string(&schema).expect("serialize OK");
    let restored: ObservationSchema = serde_json::from_str(&json).expect("deserialize OK");
    assert_eq!(schema, restored);
}

#[test]
fn action_schema_serde_roundtrip() {
    let schema = make_action_schema(ActionSemantics::NormalizedPosition, 7);
    let json = serde_json::to_string(&schema).expect("serialize OK");
    let restored: ActionSchema = serde_json::from_str(&json).expect("deserialize OK");
    assert_eq!(schema, restored);
}

#[test]
fn action_semantics_roundtrip_all_variants() {
    for sem in [
        ActionSemantics::NormalizedPosition,
        ActionSemantics::AbsoluteJointPosition,
        ActionSemantics::JointVelocity,
        ActionSemantics::Torque,
    ] {
        let json = serde_json::to_string(&sem).expect("serialize OK");
        let restored: ActionSemantics = serde_json::from_str(&json).expect("deserialize OK");
        assert_eq!(sem, restored);
    }
}

#[test]
fn frame_schema_serde_roundtrip() {
    for enc in [
        FrameEncoding::Json,
        FrameEncoding::Cdr,
        FrameEncoding::RawBytes,
        FrameEncoding::ProtobufFqn("pkg.Msg".into()),
    ] {
        let schema = FrameSchema {
            channel: "/topic".into(),
            message_type: "Msg".into(),
            encoding: enc,
            version: FrameSchema::SCHEMA_VERSION,
        };
        let json = serde_json::to_string(&schema).expect("serialize OK");
        let restored: FrameSchema = serde_json::from_str(&json).expect("deserialize OK");
        assert_eq!(schema, restored);
    }
}

#[test]
fn recorder_schema_serde_roundtrip() {
    let schema = make_recorder_schema();
    let json = serde_json::to_string(&schema).expect("serialize OK");
    let restored: RecorderSchema = serde_json::from_str(&json).expect("deserialize OK");
    assert_eq!(schema, restored);
    let names: Vec<&str> = restored
        .channels
        .iter()
        .map(|c| c.channel.as_str())
        .collect();
    assert_eq!(names, vec!["/joints", "/camera/front", "/camera/wrist"]);
}

#[test]
fn recorder_schema_validate_against_self_is_ok() {
    let schema = make_recorder_schema();
    assert!(schema.validate_against(&schema).is_ok());
}

#[test]
fn recorder_schema_validate_against_missing_channel_is_err() {
    let schema_a = make_recorder_schema();
    let mut schema_b = schema_a.clone();
    schema_b.channels.pop(); // drop /camera/wrist
    let err = schema_a.validate_against(&schema_b).unwrap_err();
    assert!(matches!(err, SchemaMismatch::ChannelSetMismatch { .. }));
}

#[test]
fn action_schema_validate_against_semantics_mismatch_is_err() {
    let schema_a = make_action_schema(ActionSemantics::NormalizedPosition, 6);
    let schema_b = make_action_schema(ActionSemantics::Torque, 6);
    let err = schema_a.validate_against(&schema_b).unwrap_err();
    assert!(matches!(
        err,
        SchemaMismatch::ActionSemanticsMismatch { .. }
    ));
}
