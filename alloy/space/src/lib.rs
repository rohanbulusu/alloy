//! Structs for describing physical space.
//! 
//! This crate includes objects from linear algebra like `Vector`s and
//! `Matrix` representations. It also carries encapsulations of
//! angular data through the `Angle` struct.
//!
//! Points of arbitrary dimension are also represented; support is
//! given particularly for `Point2` and `Point3`.

#![warn(missing_docs)]
#![deny(rust_2018_idioms)]

pub mod linal;
pub mod points;

pub use linal::*;
pub use points::*;
