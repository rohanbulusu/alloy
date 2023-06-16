//! Angle, Point2, and Point3 implementations.

use crate::linal::Vector;

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

/// Representation of a two-dimensional point.
pub struct Point2 {
	/// Horizontal component of the `Point2`
	pub x: f32,
	/// Vertical component of the `Point2`
	pub y: f32
}

impl Point2 {

	/// Constructs a new `Point2` from a two-dimensional [`Vector`].
	pub fn new(x: f32, y: f32) -> Self {
		Self { x, y }
	}

	/// Constructs a new `Point2` from a length-two array.
	pub fn with_array(coordinates: [f32; 2]) -> Self {
		Self {
			x: coordinates[0],
			y: coordinates[1]
		}
	}

	/// Constructs a new `Point2` from a length-two slice.
	pub fn with_slice<'a>(coordinates: &'a [f32; 2]) -> Self {
		Self {
			x: coordinates[0],
			y: coordinates[1]
		}
	}

	/// Constructs a new `Point2` from a [`Vec`] of length greater than or
	/// equal to two.
	///
	/// # Panics
	/// If the `Vec` provided has too small a length (less than two elements
	/// long), a panic is issued.
	/// ```should_panic
	/// # use crate::space::points::Point2;
	/// let _ = Point2::with_vec(vec![1.0]);
	/// ```
	///
	/// If the `Vec` provided has a length greater than two, then only the
	/// first two elements are used to create the `Point2`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point2;
	/// let v = Point2::with_vec(vec![1.0, 2.0]);
	/// assert_eq!(v.x, 1.0);
	/// assert_eq!(v.y, 2.0);
	/// ```
	/// ```
	/// # use crate::space::points::Point2;
	/// let v = Point2::with_vec(vec![1.0, 2.0, 3.0]);
	/// assert_eq!(v.x, 1.0);
	/// assert_eq!(v.y, 2.0);
	/// ```
	pub fn with_vec(coordinates: Vec<f32>) -> Self {
		if coordinates.len() < 2 {
			panic!("Vec must be at least two-dimensional to construct a Point2")
		}
		Self {
			x: coordinates[0],
			y: coordinates[1]
		}
	}

	/// Constructs a new `Point2` from a [`Vector`] of dimension greater than 
	/// or equal to two.
	///
	/// # Panics
	/// If the `Vector` provided has a dimension less than or equal to two, a
	/// panic is issued.
	/// ```should_panic
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point2;
	/// let _ = Point2::with_vector(Vector::new([1.0]));
	/// ```
	///
	/// If the `Vector` provided has a length greater than two, then only the
	/// first two elements are used to create the `Point2`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point2;
	/// let v = Point2::with_vector(Vector::new([1.0, 2.0]));
	/// assert_eq!(v.x, 1.0);
	/// assert_eq!(v.y, 2.0);
	/// ```
	/// ```
	///	# use crate::space::linal::Vector;
	/// # use crate::space::points::Point2;
	/// let v = Point2::with_vector(Vector::new([1.0, 2.0, 3.0]));
	/// assert_eq!(v.x, 1.0);
	/// assert_eq!(v.y, 2.0);
	/// ```
	pub fn with_vector(coordinates: Vector<f32>) -> Self {
		if coordinates.dim < 2 {
			panic!("Vector must be at least two-dimensional to construct a Point2")
		}
		Self {
			x: coordinates.get(0),
			y: coordinates.get(1)
		}
	}

	/// Translates `self` by `translation` in place.
	///
	/// If the provided [`Vector`] is zero-dimensional, `self` is directly 
	/// returned; if it's one-dimensional, `translation` is treated like a 
	/// two-dimensional vector with horizontal component specified by the
	/// provided single element; if it's of dimension greater than two, the 
	/// first two components are used as the transformation elements.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point2;
	/// let v = Vector::new([]);
	/// let position = Point2::new(0.0, 0.0).translate(v);
	/// assert_eq!(position.x, 0.0);
	/// assert_eq!(position.y, 0.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point2;
	/// let v = Vector::new([1.0]);
	/// let position = Point2::new(0.0, 0.0).translate(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 0.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point2;
	/// let v = Vector::new([1.0, 2.0]);
	/// let position = Point2::new(0.0, 0.0).translate(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 2.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point2;
	/// let v = Vector::new([1.0, 2.0, 3.0]);
	/// let position = Point2::new(0.0, 0.0).translate(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 2.0);
	/// ```
	pub fn translate(mut self, translation: Vector<f32>) -> Self {
		if translation.dim == 0 {
			return self;
		}
		if translation.dim == 1 {
			self.x += translation.get(0);
			return self;
		}
		self.x += translation.get(0);
		self.y += translation.get(1);
		self
	}

	/// Computes the distance between `self` and `other`.
	pub fn dist(&self, other: &Self) -> f32 {
		let x = (self.x - other.x).abs();
		let y = (self.y - other.y).abs();
		(x*x + y*y).sqrt()
	}

}
