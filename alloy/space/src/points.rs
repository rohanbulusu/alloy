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
#[derive(Clone, Copy)]
pub struct Point2 {
	/// Horizontal component of the `Point2`
	pub x: f32,
	/// Vertical component of the `Point2`
	pub y: f32
}

impl Point2 {

	/// Constructs a new `Point2` from individual [`f32`] components.
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
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point2;
	/// let a = Point2::new(0.0, 0.0);
	/// let b = Point2::new(-4.0, 3.0);
	/// assert_eq!(a.to(&b), 5.0);
	/// ```
	/// ```
	/// # use crate::space::points::Point2;
	/// let a = Point2::new(1.0, 3.0);
	/// let b = Point2::new(-3.0, 6.0);
	/// assert_eq!(a.to(&b), 5.0);
	/// ```
	/// ```
	/// # use crate::space::points::Point2;
	/// let a = Point2::new(4.0, 6.0);
	/// let b = Point2::new(4.0, 6.0);
	/// assert_eq!(a.to(&b), 0.0);
	/// ```
	pub fn to(&self, other: &Self) -> f32 {
		let x = (self.x - other.x).abs();
		let y = (self.y - other.y).abs();
		(x*x + y*y).sqrt()
	}

	/// Takes the line defined by `a` and `b` and tests whether or not `self`
	/// lies on it.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point2;
	/// let a = Point2::new(3.0, 2.0);
	/// let b = Point2::new(4.0, 1.0);
	/// let c = Point2::new(1.0, 4.0);
	/// assert!(c.collinear_with(&a, &b));
	/// ```
	/// ```
	/// # use crate::space::points::Point2;
	/// let a = Point2::new(3.0, 2.0);
	/// let b = Point2::new(4.0, 1.0);
	/// let c = Point2::new(2.0, -9.0);
	/// assert!(!c.collinear_with(&a, &b));
	/// ```
	pub fn collinear_with(&self, a: &Self, b: &Self) -> bool {
		if a == b {
			return true;
		}
		// if the a and b form a vertical line, the slope is
		// undefined, so this just handles that exceptional case
		if a.x == b.x {
			return self.x == a.x;
		}
		let slope = (a.y - b.y) / (a.x - b.x);
		return self.y - a.y == slope * (self.x - a.x);
	}

	/// Whether or not `self` is to the left of `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point2;
	/// let a = Point2::new(-3.0, 2.0);
	/// let b = Point2::new(1.0, 5.0);
	/// assert!(a.left_of(&b));
	/// assert!(!b.left_of(&a));
	/// ```
	pub fn left_of(&self, other: &Self) -> bool {
		self.x < other.x
	}

	/// Whether or not `self` is to the right of `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point2;
	/// let a = Point2::new(2.0, 1.0);
	/// let b = Point2::new(-3.0, 2.0);
	/// assert!(a.right_of(&b));
	/// assert!(!b.right_of(&a));
	/// ```
	pub fn right_of(&self, other: &Self) -> bool {
		self.x > other.x
	}

	/// Whether or not `self` is above `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point2;
	/// let a = Point2::new(-3.0, 5.0);
	/// let b = Point2::new(1.0, 2.0);
	/// assert!(a.above(&b));
	/// assert!(!b.above(&a));
	/// ```
	pub fn above(&self, other: &Self) -> bool {
		self.y > other.y
	}

	/// Whether or not `self` is below `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point2;
	/// let a = Point2::new(-3.0, 2.0);
	/// let b = Point2::new(1.0, 5.0);
	/// assert!(a.below(&b));
	/// assert!(!b.below(&a));
	/// ```
	pub fn below(&self, other: &Self) -> bool {
		self.y < other.y
	}

	/// Whether or not `self` is vertically aligned with `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point2;
	/// let a = Point2::new(1.0, 2.0);
	/// let b = Point2::new(1.0, 5.0);
	/// assert!(a.vertical_aligned(&b));
	/// ```
	pub fn vertical_aligned(&self, other: &Self) -> bool {
		self.x == other.x
	}

	/// Whether or not `self` is horizontally aligned with `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point2;
	/// let a = Point2::new(3.0, 2.0);
	/// let b = Point2::new(-4.0, 2.0);
	/// assert!(a.horizontal_aligned(&b));
	/// ```
	pub fn horizontal_aligned(&self, other: &Self) -> bool {
		self.y == other.y
	}

	/// Returns the relative positioning of `other` with respect to `self`.
	///
	/// This is primarily to enable pattern matching on the
	/// [`RelativeLocation`] enum, not to guage a specific position. If a 
	/// particular question of positioning is needed, the use of 
	/// [`Point2::left_of`], [`Point2::right_of`], [`Point2::above`], or
	/// [`Point2::below`] is recommended.
	pub fn compare(&self, other: &Self) -> RelativeLocation {
		if self.left_of(other) {
			return RelativeLocation::Left;
		}
		if self.right_of(other) {
			return RelativeLocation::Right;
		}
		if self.above(other) {
			return RelativeLocation::Above;
		}
		if self.below(other) {
			return RelativeLocation::Below;
		}
		RelativeLocation::Overlap
	}

	/// Determines the slope of the line between `a` and `b`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point2;
	/// let a = Point2::new(1.0, 2.0);
	/// let b = Point2::new(2.0, 3.0);
	/// assert_eq!(Point2::slope(&a, &b), 1.0);
	/// ```
	pub fn slope(a: &Self, b: &Self) -> f32 {
		(a.y - b.y) / (a.x - b.x)
	}

	/// Constructs a linear equation given two `Point2`s.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point2;
	/// let a = Point2::new(1.0, 2.0);
	/// let b = Point2::new(2.0, 3.0);
	/// let y = Point2::line(a, b);
	/// assert_eq!(y(1.0), 2.0);
	/// assert_eq!(y(2.0), 3.0);
	/// assert_eq!(y(3.0), 4.0);
	/// ```
	pub fn line(a: Self, b: Self) -> impl Fn(f32) -> f32 {
		return move |x| Self::slope(&a, &b) * (x - a.x) + a.y;
	}

}

impl PartialEq for Point2 {
	fn eq(&self, other: &Self) -> bool {
		let x_equality = (self.x - other.x).abs() < f32::EPSILON;
		let y_equality = (self.y - other.y).abs() < f32::EPSILON;
		x_equality && y_equality
	}
}

/// Representation of a three-dimensional point.
#[derive(Clone, Copy)]
pub struct Point3 {
	/// Horizontal component of the `Point3`
	pub x: f32,
	/// Vertical component of the `Point3`
	pub y: f32,
	/// Depth component of the `Point3`
	pub z: f32
}

impl Point3 {

	/// Constructs a new `Point3` from individual [`f32`] components.
	pub fn new(x: f32, y: f32, z: f32) -> Self {
		Self { x, y, z }
	}

	/// Constructs a new `Point3` from a length-three array.
	pub fn with_array(coordinates: [f32; 3]) -> Self {
		Self {
			x: coordinates[0],
			y: coordinates[1],
			z: coordinates[2]
		}
	}

	/// Constructs a new `Point3` from a length-three slice.
	pub fn with_slice<'a>(coordinates: &'a [f32; 3]) -> Self {
		Self {
			x: coordinates[0],
			y: coordinates[1],
			z: coordinates[2]
		}
	}

	/// Constructs a new `Point3` from a [`Vec`] of length greater than or
	/// equal to three.
	///
	/// # Panics
	/// If the `Vec` provided has too small a length (less than three elements
	/// lone), a panic is issued.
	/// ```should_panic
	/// # use crate::space::points::Point3;
	/// let _ = Point3::with_vec(vec![1.0]);
	/// ```
	/// ```should_panic
	/// # use crate::space::points::Point3;
	/// let _ = Point3::with_vec(vec![1.0, 2.0]);
	/// ```
	/// If the `Vec` provided has a length greater than three, then only the
	/// first three elments are used to create the `Point3`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point3;
	/// let v = Point3::with_vec(vec![1.0, 2.0, 3.0]);
	/// assert_eq!(v.x, 1.0);
	/// assert_eq!(v.y, 2.0);
	/// assert_eq!(v.z, 3.0);
	/// ```
	/// ```
	/// # use crate::space::points::Point3;
	/// let v = Point3::with_vec(vec![1.0, 2.0, 3.0, 4.0]);
	/// assert_eq!(v.x, 1.0);
	/// assert_eq!(v.y, 2.0);
	/// assert_eq!(v.z, 3.0);
	/// ```
	pub fn with_vec(coordinates: Vec<f32>) -> Self {
		if coordinates.len() < 3 {
			panic!("Cannot construct a Point3 from a Vec with less than three elements")
		}
		Self {
			x: coordinates[0],
			y: coordinates[1],
			z: coordinates[2]
		}
	}

	/// Constructs a new `Point3` from a [`Vector`] of length greater than or
	/// equal to three.
	///
	/// # Panics
	/// If the `Vector` provided has too small a length (less than three elements
	/// lone), a panic is issued.
	/// ```should_panic
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let _ = Point3::with_vector(Vector::new([1.0]));
	/// ```
	/// ```should_panic
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let _ = Point3::with_vector(Vector::new([1.0, 2.0]));
	/// ```
	/// If the `Vector` provided has a length greater than three, then only the
	/// first three elments are used to create the `Point3`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let v = Point3::with_vector(Vector::new([1.0, 2.0, 3.0]));
	/// assert_eq!(v.x, 1.0);
	/// assert_eq!(v.y, 2.0);
	/// assert_eq!(v.z, 3.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let v = Point3::with_vector(Vector::new([1.0, 2.0, 3.0, 4.0]));
	/// assert_eq!(v.x, 1.0);
	/// assert_eq!(v.y, 2.0);
	/// assert_eq!(v.z, 3.0);
	/// ```
	pub fn with_vector(coordinates: Vector<f32>) -> Self {
		if coordinates.dim < 3 {
			panic!("Cannot construct a Point3 from a Vector with less than three indices")
		}
		Self {
			x: coordinates.get(0),
			y: coordinates.get(1),
			z: coordinates.get(2)
		}
	}

}

/// Specifies relative locations between [`Point2`]s and/or [`Point3`]s.
///
/// This is subject to change depending on the need for the specification of
/// horizontal and vertical coordinate equality.
#[non_exhaustive]
pub enum RelativeLocation {
	/// For `Point3`s only, if one is in front of the other
	Before,
	/// For `Point3`s only, f one is behind the other
	Behind,
	/// If one is above the other
	Above,
	/// If one is below the other
	Below,
	/// If one is to the left of the other
	Left,
	/// If one is to the right of the other
	Right,
	/// If both points are the same
	Overlap
}

#[cfg(test)]
mod point2 {

	use super::Point2;

	mod collinearity {

		use super::Point2;

		#[test]
		fn all_same() {
			let a = Point2::new(1.0, 3.0);
			let b = Point2::new(1.0, 3.0);
			let c = Point2::new(1.0, 3.0);
			assert!(c.collinear_with(&a, &b))
		}

		#[test]
		fn overlapping_endpoints() {
			let a = Point2::new(1.0, 3.0);
			let b = Point2::new(1.0, 3.0);
			let c = Point2::new(4.0, 7.0);
			assert!(c.collinear_with(&a, &b))
		}

		#[test]
		fn point_overlaps_with_left_endpoint() {
			let a = Point2::new(4.0, 7.0);
			let b = Point2::new(1.0, 3.0);
			let c = Point2::new(1.0, 3.0);
			assert!(c.collinear_with(&a, &b))
		}

		#[test]
		fn point_overlaps_with_right_endpoint() {
			let a = Point2::new(1.0, 3.0);
			let b = Point2::new(4.0, 7.0);
			let c = Point2::new(1.0, 3.0);
			assert!(c.collinear_with(&a, &b))
		}

		#[test]
		fn vertical() {
			let a = Point2::new(1.0, 3.0);
			let b = Point2::new(1.0, -4.0);
			let c = Point2::new(1.0, 4.3);
			assert!(c.collinear_with(&a, &b))
		}

		#[test]
		fn horizontal() {
			let a = Point2::new(3.0, 1.0);
			let b = Point2::new(-4.0, 1.0);
			let c = Point2::new(4.3, 1.0);
			assert!(c.collinear_with(&a, &b))
		}

	}

}
