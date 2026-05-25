//! Proves that the new Action accessors never panic on non-Continuous variants.
//!
//! Zero `#[should_panic]` attributes — the entire point of W3 PR1 is that
//! the panics are gone; every assertion is structural (`is_none`, `is_err`,
//! `matches!`, `assert_eq!`).

use clankers_core::error::ActionKindError;
use clankers_core::types::Action;

#[test]
fn as_continuous_returns_none_for_discrete() {
    let action = Action::Discrete(0);
    assert!(
        action.as_continuous().is_none(),
        "must not panic, must return None"
    );
}

#[test]
fn as_continuous_returns_none_for_multi_discrete() {
    let action = Action::MultiDiscrete(vec![1, 2]);
    assert!(action.as_continuous().is_none());
}

#[test]
fn as_continuous_returns_some_for_continuous() {
    let action = Action::Continuous(vec![0.5, -0.5]);
    assert_eq!(action.as_continuous(), Some(&[0.5_f32, -0.5_f32][..]));
}

#[test]
fn try_into_continuous_returns_err_for_discrete() {
    let action = Action::Discrete(0);
    let err = action.try_into_continuous().unwrap_err();
    assert!(matches!(err, ActionKindError::ExpectedContinuous { .. }));
}

#[test]
fn try_into_continuous_returns_err_for_multi_discrete() {
    let action = Action::MultiDiscrete(vec![1, 2]);
    assert!(action.try_into_continuous().is_err());
}

#[test]
fn try_into_continuous_returns_ok_for_continuous() {
    let action = Action::Continuous(vec![1.0, 2.0]);
    assert_eq!(action.try_into_continuous().unwrap(), vec![1.0, 2.0]);
}
