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
	#[inline]
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
	#[inline]
	pub fn to_radians(theta: f64) -> f64 {
		theta * Self::DEGREES_TO_RADIANS
	}

	/// Constructs a new `Angle` from a degrees value.
	#[inline]
	pub fn with_degrees(theta: f64) -> Self {
		Self {
			degrees: theta,
			radians: Self::to_radians(theta)
		}
	}

	/// Constructs a new `Angle` from a radians value.
	#[inline]
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
	#[inline]
	pub const fn new(x: f32, y: f32) -> Self {
		Self { x, y }
	}

	/// Constructs a new `Point2` from a length-two array.
	#[inline]
	pub const fn with_array(coordinates: [f32; 2]) -> Self {
		Self {
			x: coordinates[0],
			y: coordinates[1]
		}
	}

	/// Constructs a new `Point2` from a length-two slice.
	#[inline]
	pub const fn with_slice<'a>(coordinates: &'a [f32; 2]) -> Self {
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
	#[inline]
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
	#[inline]
	pub fn with_vector(coordinates: Vector<f32>) -> Self {
		if coordinates.dim < 2 {
			panic!("Vector must be at least two-dimensional to construct a Point2")
		}
		Self {
			x: coordinates.get(0),
			y: coordinates.get(1)
		}
	}

	/// Performs an affine transformation of `self` by `translation`.
	///
	/// If the provided [`Vector`] is zero-dimensional, `self` is left
	/// unchanged; if it's one-dimensional, `translation` is treated like a 
	/// two-dimensional vector with horizontal component specified by the
	/// provided single element; if it's of dimension greater than two, the 
	/// first two components are used as the transformation elements.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point2;
	/// let v = Vector::new([]);
	/// let mut position = Point2::new(0.0, 0.0);
	/// position.slide_by(v);
	/// assert_eq!(position.x, 0.0);
	/// assert_eq!(position.y, 0.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point2;
	/// let v = Vector::new([1.0]);
	/// let mut position = Point2::new(0.0, 0.0);
	/// position.slide_by(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 0.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point2;
	/// let v = Vector::new([1.0, 2.0]);
	/// let mut position = Point2::new(0.0, 0.0);
	/// position.slide_by(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 2.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point2;
	/// let v = Vector::new([1.0, 2.0, 3.0]);
	/// let mut position = Point2::new(0.0, 0.0);
	/// position.slide_by(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 2.0);
	/// ```
	pub fn slide_by(&mut self, translation: Vector<f32>) {
		if translation.dim == 0 {
			return;
		}
		if translation.dim == 1 {
			self.x += translation.get(0);
			return;
		}
		self.x += translation.get(0);
		self.y += translation.get(1);
	}

	/// Performs an affine transformation of `self` by `translation` in place.
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
	#[inline]
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
	#[inline]
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
	#[inline]
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
	#[inline]
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
	#[inline]
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
	#[inline]
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
	#[inline]
	pub fn horizontal_aligned(&self, other: &Self) -> bool {
		self.y == other.y
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
	#[inline]
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
	#[inline]
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
	#[inline]
	pub fn new(x: f32, y: f32, z: f32) -> Self {
		Self { x, y, z }
	}

	/// Constructs a new `Point3` from a length-three array.
	#[inline]
	pub fn with_array(coordinates: [f32; 3]) -> Self {
		Self {
			x: coordinates[0],
			y: coordinates[1],
			z: coordinates[2]
		}
	}

	/// Constructs a new `Point3` from a length-three slice.
	#[inline]
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
	#[inline]
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
	#[inline]
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

	/// Performs an affine transformation of `self`.
	///
	/// If the provided [`Vector`] is zero-dimensional, `self` is left
	/// unchanged; if it's one-dimensional, `translation` is treated like a 
	/// two-dimensional vector with horizontal component specified by the
	/// provided single element; if it's of dimension two, the 
	/// two components are used as horizontal and vertical transformation 
	/// elements respectively. For `Vector`s of dimension three or greater, the
	/// first three elements only are used in the transformation.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let v = Vector::new([]);
	/// let mut position = Point3::new(0.0, 0.0, 0.0);
	/// position.slide_by(v);
	/// assert_eq!(position.x, 0.0);
	/// assert_eq!(position.y, 0.0);
	/// assert_eq!(position.z, 0.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let v = Vector::new([1.0]);
	/// let mut position = Point3::new(0.0, 0.0, 0.0);
	/// position.slide_by(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 0.0);
	/// assert_eq!(position.z, 0.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let v = Vector::new([1.0, 2.0]);
	/// let mut position = Point3::new(0.0, 0.0, 0.0);
	/// position.slide_by(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 2.0);
	/// assert_eq!(position.z, 0.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let v = Vector::new([1.0, 2.0, 3.0]);
	/// let mut position = Point3::new(0.0, 0.0, 0.0);
	/// position.slide_by(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 2.0);
	/// assert_eq!(position.z, 3.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let v = Vector::new([1.0, 2.0, 3.0, 4.0]);
	/// let mut position = Point3::new(0.0, 0.0, 0.0);
	/// position.slide_by(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 2.0);
	/// assert_eq!(position.z, 3.0);
	/// ```
	pub fn slide_by(&mut self, translation: Vector<f32>) {
		if translation.dim == 0 {
			return;
		}
		if translation.dim == 1 {
			self.x += translation.get(0);
			return;
		}
		if translation.dim == 2 {
			self.x += translation.get(0);
			self.y += translation.get(1);
			return;
		}
		self.x += translation.get(0);
		self.y += translation.get(1);
		self.z += translation.get(2);
	}

	/// Performs an affine transformation of `self` in place.
	///
	/// If the provided [`Vector`] is zero-dimensional, `self` is left
	/// unchanged; if it's one-dimensional, `translation` is treated like a 
	/// two-dimensional vector with horizontal component specified by the
	/// provided single element; if it's of dimension two, the 
	/// two components are used as horizontal and vertical transformation 
	/// elements respectively. For `Vector`s of dimension three or greater, the
	/// first three elements only are used in the transformation.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let v = Vector::new([]);
	/// let mut position = Point3::new(0.0, 0.0, 0.0).translate(v);
	/// assert_eq!(position.x, 0.0);
	/// assert_eq!(position.y, 0.0);
	/// assert_eq!(position.z, 0.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let v = Vector::new([1.0]);
	/// let mut position = Point3::new(0.0, 0.0, 0.0).translate(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 0.0);
	/// assert_eq!(position.z, 0.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let v = Vector::new([1.0, 2.0]);
	/// let mut position = Point3::new(0.0, 0.0, 0.0).translate(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 2.0);
	/// assert_eq!(position.z, 0.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let v = Vector::new([1.0, 2.0, 3.0]);
	/// let mut position = Point3::new(0.0, 0.0, 0.0).translate(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 2.0);
	/// assert_eq!(position.z, 3.0);
	/// ```
	/// ```
	/// # use crate::space::linal::Vector;
	/// # use crate::space::points::Point3;
	/// let v = Vector::new([1.0, 2.0, 3.0, 4.0]);
	/// let mut position = Point3::new(0.0, 0.0, 0.0).translate(v);
	/// assert_eq!(position.x, 1.0);
	/// assert_eq!(position.y, 2.0);
	/// assert_eq!(position.z, 3.0);
	/// ```
	pub fn translate(mut self, translation: Vector<f32>) -> Self {
		if translation.dim == 1 {
			self.x += translation.get(0);
		}
		if translation.dim == 2 {
			self.x += translation.get(0);
			self.y += translation.get(1);
		}
		if translation.dim >= 3 {
			self.x += translation.get(0);
			self.y += translation.get(1);
			self.z += translation.get(2);
		}
		self
	}

	/// Computes the distance between `self` and `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(0.0, 0.0, 0.0);
	/// let b = Point3::new(1.0, 2.0, 2.0);
	/// assert_eq!(a.to(&b), 3.0);
	/// ```
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(0.0, 0.0, 0.0);
	/// let b = Point3::new(1.0, -2.0, 2.0);
	/// assert_eq!(a.to(&b), 3.0);
	/// ```
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(0.0, 1.0, 0.0);
	/// let b = Point3::new(1.0, -1.0, 2.0);
	/// assert_eq!(a.to(&b), 3.0);
	/// ```
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(1.0, -1.0, 2.0);
	/// let b = Point3::new(1.0, -1.0, 2.0);
	/// assert_eq!(a.to(&b), 0.0);
	/// ```
	#[inline]
	pub fn to(&self, other: &Self) -> f32 {
		let x = (self.x - other.x).abs();
		let y = (self.y - other.y).abs();
		let z = (self.z - other.z).abs();
		(x*x + y*y + z*z).sqrt()
	}

	/// Constructs a linear equation given two `Point3`s.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(1.0, 2.0, 3.0);
	/// let b = Point3::new(2.0, 3.0, 3.0);
	/// let y = Point3::line(a, b);
	/// assert_eq!(y(1.0, 2.0), 3.0);
	/// assert_eq!(y(2.0, 3.0), 3.0);
	/// assert_eq!(y(3.0, 4.0), 3.0);
	/// ```
	#[inline]
	pub fn line(a: Self, b: Self) -> impl Fn(f32, f32) -> f32 {
		let x_slope = a.x - b.x;
		let y_slope = a.y - b.y;
		let z_slope = a.z - b.z;
		return move |x, y| {
			let x_term = z_slope*(x - a.x) / x_slope;
			let y_term = z_slope*(y - a.y) / y_slope;
			x_term - y_term + a.z
		};
	}

	/// Takes the line defined by `a` and `b` and tests whether or not `self`
	/// lies on it.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(3.0, 2.0, 3.0);
	/// let b = Point3::new(4.0, 1.0, 3.0);
	/// let c = Point3::new(1.0, 4.0, 3.0);
	/// assert!(c.collinear_with(&a, &b));
	/// ```
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(4.0, 2.0, 3.0);
	/// let b = Point3::new(3.0, 1.0, 3.0);
	/// let c = Point3::new(5.0, -9.0, 3.0);
	/// assert!(!c.collinear_with(&a, &b));
	/// ```
	pub fn collinear_with(&self, a: &Self, b: &Self) -> bool {
		if a == b {
			return true;
		}
		if a.x == b.x {
			let a_without_x = Point2::new(a.y, a.z);
			let b_without_x = Point2::new(b.y, b.z);
			let self_without_x = Point2::new(self.y, self.z);
			return self_without_x.collinear_with(&a_without_x, &b_without_x);
		}
		if a.y == b.y {
			let a_without_y = Point2::new(a.x, a.z);
			let b_without_y = Point2::new(b.x, b.z);
			let self_without_y = Point2::new(self.x, self.z);
			return self_without_y.collinear_with(&a_without_y, &b_without_y);
		}
		if a.z == b.z {
			let a_without_z = Point2::new(a.x, a.y);
			let b_without_z = Point2::new(b.x, b.y);
			let self_without_z = Point2::new(self.x, self.y);
			return self_without_z.collinear_with(&a_without_z, &b_without_z);
		}
		Point3::line(*a, *b)(self.x, self.y) == self.z
	}

	/// Whether or not `self` is left of `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(-4.0, 2.0, 1.0);
	/// let b = Point3::new(3.0, 4.0, 2.0);
	/// assert!(a.left_of(&b));
	/// assert!(!b.left_of(&a));
	/// ```
	#[inline]
	pub fn left_of(&self, other: &Self) -> bool {
		self.x < other.x
	}


	/// Whether or not `self` is right of `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(3.0, 4.0, 2.0);
	/// let b = Point3::new(-4.0, 2.0, 1.0);
	/// assert!(a.right_of(&b));
	/// assert!(!b.right_of(&a));
	/// ```
	#[inline]
	pub fn right_of(&self, other: &Self) -> bool {
		self.x > other.x
	}


	/// Whether or not `self` is above `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(3.0, 4.0, 2.0);
	/// let b = Point3::new(-2.0, 2.0, 1.0);
	/// assert!(a.above(&b));
	/// assert!(!b.above(&a));
	/// ```
	#[inline]
	pub fn above(&self, other: &Self) -> bool {
		self.y > other.y
	}


	/// Whether or not `self` is below `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(3.0, 2.0, 1.0);
	/// let b = Point3::new(-2.0, 4.0, 2.0);
	/// assert!(a.below(&b));
	/// assert!(!b.below(&a));
	/// ```
	#[inline]
	pub fn below(&self, other: &Self) -> bool {
		self.y < other.y
	}

	/// Whether or not `self` is horizontally aligned with `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(3.0, 2.0, 4.0);
	/// let b = Point3::new(3.0, -2.0, -5.0);
	/// assert!(a.horizontal_aligned(&b));
	/// assert!(b.horizontal_aligned(&a));
	/// ```
	#[inline]
	pub fn horizontal_aligned(&self, other: &Self) -> bool {
		self.x == other.x
	}

	/// Whether or not `self` is vertically aligned with `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(2.0, -2.0, -5.0);
	/// let b = Point3::new(-5.0, -2.0, 3.0);
	/// assert!(a.vertical_aligned(&b));
	/// assert!(b.vertical_aligned(&a));
	/// ```
	#[inline]
	pub fn vertical_aligned(&self, other: &Self) -> bool {
		self.y == other.y
	}

	/// Whether or not `self` is aligned along the z-axis with `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(3.0, -2.0, -5.0);
	/// let b = Point3::new(4.0, 2.0, -5.0);
	/// assert!(a.depth_aligned(&b));
	/// assert!(b.depth_aligned(&a));
	/// ```
	#[inline]
	pub fn depth_aligned(&self, other: &Self) -> bool {
		self.z == other.z
	}

	/// Determines the unique plane described by `a`, `b`, and `c`.
	///
	/// # Panics
	/// If `a`, `b`, and `c` are collinear, it's not possible to construct a 
	/// plane; this yields a panic.
	/// ```should_panic
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(1.0, 0.0, 2.0);
	/// let b = Point3::new(2.0, 0.0, 3.0);
	/// let c = Point3::new(3.0, 0.0, 4.0);
	/// let _ = Point3::plane(a, b, c);
	/// ```
	/// z = x - 2y + 2
	/// # Examples
	/// ```
	/// # use crate::space::points::Point3;
	/// let a = Point3::new(2.0, 1.0, 2.0);
	/// let b = Point3::new(1.0, 5.0, -7.0);
	/// let c = Point3::new(0.0, 0.0, 2.0);
	/// let p = Point3::plane(a, b, c);
	/// assert_eq!(p(1.0, 5.0), -7.0);
	/// assert_eq!(p(2.0, 1.0), 2.0);
	/// assert_eq!(p(0.0, 0.0), 2.0);
	/// assert_eq!(p(12.0, 3.0), 8.0);
	/// ```
	pub fn plane(a: Self, b: Self, c: Self) -> impl Fn(f32, f32) -> f32 {
		if a.collinear_with(&b, &c) {
			panic!("Points must be collinear to form a unique plane")
		}
		let vec_one = Vector::new([
			a.x - b.x,
			a.y - b.y,
			a.z - b.z
		]);
		let vec_two = Vector::new([
			b.x - c.x,
			b.y - c.y,
			b.z - c.z
		]);
		let normal = Vector::cross(vec_one, vec_two);
		let intercept = -(normal.get(0)*a.x + normal.get(1)*a.y + normal.get(2)*a.z);
		return move |x, y| {
			let unscaled = x*normal.get(0) + y*normal.get(1) + intercept;
			unscaled / -normal.get(2)
		}
	}

}

impl PartialEq for Point3 {
	fn eq(&self, other: &Self) -> bool {
		let x_equality = (self.x - other.x).abs() < f32::EPSILON;
		let y_equality = (self.y - other.y).abs() < f32::EPSILON;
		let z_equality = (self.z - other.z).abs() < f32::EPSILON;
		x_equality && y_equality && z_equality
	}
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

#[cfg(test)]
mod point3 {

	use super::Point3;

	mod collinear {

		use super::Point3;

		mod in_plane {

			use super::Point3;

			#[test]
			fn all_same() {
				let a = Point3::new(1.0, 3.0, 0.0);
				let b = Point3::new(1.0, 3.0, 0.0);
				let c = Point3::new(1.0, 3.0, 0.0);
				assert!(c.collinear_with(&a, &b))
			}

			#[test]
			fn overlapping_endpoints() {
				let a = Point3::new(1.0, 3.0, 0.0);
				let b = Point3::new(1.0, 3.0, 0.0);
				let c = Point3::new(4.0, 7.0, 0.0);
				assert!(c.collinear_with(&a, &b))
			}

			#[test]
			fn point_overlaps_with_left_endpoint() {
				let a = Point3::new(4.0, 7.0, 0.0);
				let b = Point3::new(1.0, 3.0, 0.0);
				let c = Point3::new(1.0, 3.0, 0.0);
				assert!(c.collinear_with(&a, &b))
			}

			#[test]
			fn point_overlaps_with_right_endpoint() {
				let a = Point3::new(1.0, 3.0, 0.0);
				let b = Point3::new(4.0, 7.0, 0.0);
				let c = Point3::new(1.0, 3.0, 0.0);
				assert!(c.collinear_with(&a, &b))
			}

			#[test]
			fn vertical() {
				let a = Point3::new(1.0, 3.0, 0.0);
				let b = Point3::new(1.0, -4.0, 0.0);
				let c = Point3::new(1.0, 4.3, 0.0);
				assert!(c.collinear_with(&a, &b))
			}

			#[test]
			fn horizontal() {
				let a = Point3::new(3.0, 1.0, 0.0);
				let b = Point3::new(-4.0, 1.0, 0.0);
				let c = Point3::new(4.3, 1.0, 0.0);
				assert!(c.collinear_with(&a, &b))
			}

		}

	}

}
