//! Angle, Point2, and Point3 implementations.

/// Container for angular information.
///
/// An `Angle` can be constructed either from a degrees value or a radians
/// value; from there, either a degrees or radians representation of the angle
/// can be requested at any time from the object.
/// 
/// ```
/// # use crate::space::points::Angle;
/// let theta = Angle::with_degrees(90.0);
/// assert_eq!(theta.degrees, 90.0);
/// assert_eq!(theta.radians, std::f64::consts::FRAC_PI_2);
/// ```
pub struct Angle {
	/// `Angle` value in degrees
	pub degrees: f64,
	/// `Angle` value in radians
	pub radians: f64
}

impl Angle {

	const RADIANS_TO_DEGREES: f64 = 180.0 / std::f64::consts::PI;

	const DEGREES_TO_RADIANS: f64 = std::f64::consts::PI / 180.0;

	/// Converts `theta` from radians to degrees.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Angle;
	/// let theta = std::f64::consts::PI;
	/// assert_eq!(Angle::to_degrees(theta), 180.0);
	/// ```
	pub fn to_degrees(theta: f64) -> f64 {
		theta * Self::RADIANS_TO_DEGREES
	}

	/// Converts `theta` from degrees to radians.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Angle;
	/// let theta = 180.0;
	/// assert_eq!(Angle::to_radians(theta), std::f64::consts::PI);
	/// ```
	pub fn to_radians(theta: f64) -> f64 {
		theta * Self::DEGREES_TO_RADIANS
	}

	/// Constructs a new `Angle` from a degrees value.
	pub fn with_degrees(theta: f64) -> Self {
		Self {
			degrees: theta,
			radians: Self::to_radians(theta)
		}
	}

	/// Constructs a new `Angle` from a radians value.
	pub fn with_radians(theta: f64) -> Self {
		Self {
			degrees: Self::to_degrees(theta),
			radians: theta
		}
	}

}
