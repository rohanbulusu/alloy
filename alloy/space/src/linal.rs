//! Linear algebra support.

use std::ops::{Add, Sub, Mul, Div, Neg};

/// Vector implementation for elements of type `T`.
pub struct Vector<T> {
	ptr: std::ptr::NonNull<T>,
	dim: usize
}

impl<T> Vector<T> {

	/// Constructs a new `Vector` from an array of components of type `T`.
	pub fn new<const N: usize>(components: [T; N]) -> Self {
		// set up the allocation
		let layout = std::alloc::Layout::array::<T>(N).unwrap();
		let allocation = unsafe { std::alloc::alloc(layout) };
		let ptr = match std::ptr::NonNull::new(allocation as *mut T) {
			Some(p) => p,
			None => std::alloc::handle_alloc_error(layout)
		};
		// set the components
		for (i, component) in components.into_iter().enumerate() {
			unsafe { std::ptr::write(ptr.as_ptr().add(i), component) }
		}
		Self {
			ptr,
			dim: N
		}
	}

	/// Constructs a new `Vector` from a [`Vec`] of components of type `T`.
	pub fn with_vec(components: Vec<T>) -> Self {
		let dim = components.len();
		// set up the allocation
		let layout = std::alloc::Layout::array::<T>(dim).unwrap();
		let allocation = unsafe { std::alloc::alloc(layout) };
		let ptr = match std::ptr::NonNull::new(allocation as *mut T) {
			Some(p) => p,
			None => std::alloc::handle_alloc_error(layout)
		};
		// set the components
		for (i, component) in components.into_iter().enumerate() {
			unsafe { std::ptr::write(ptr.as_ptr().add(i), component) }
		}
		Self {
			ptr,
			dim
		}
	}

	/// Returns the component of `self` specified by `index`.
	///
	/// # Panics
	/// If `index` is greater than or equal to the dimension of the vector,
	/// then a panic is issued.
	///
	/// ```should_panic
	/// # use crate::space::linal::Vector;
	/// let v = Vector::new([1, 2]);
	/// let _ = v.get(2);
	/// ```
	pub fn get(&self, index: usize) -> T {
		if index >= self.dim {
			panic!("Cannot fetch element with index {} for a {}-dimensional Vector", index, self.dim);
		}
		unsafe { std::ptr::read(self.ptr.as_ptr().add(index)) }
	}

}

impl<T> Vector<T> where T: Sub<Output=T> + Mul<Output=T> {

	/// Cross product between `a` and `b`.
	///
	/// # Panics
	/// `a` and `b` must both be three-dimensional; the cross product is
	/// undefined for `Vector`s of greater dimensionality.
	/// 
	/// ```should_panic
	/// # use crate::space::linal::Vector;
	/// let _ = Vector::cross(Vector::new([1.0, 2.0]), Vector::new([3.0, 4.0]));
	/// ```
	pub fn cross(a: Self, b: Self) -> Self {
		if a.dim != b.dim {
			panic!("Cross products can only be taken between Vectors of the same dimensionality")
		}
		if a.dim != 3 {
			panic!("Cross products are only defined for three-dimensional Vectors")
		}
		Vector::new([
			a.get(1)*b.get(2) - a.get(2)*b.get(1),
			a.get(2)*b.get(0) - a.get(0)*b.get(2),
			a.get(0)*b.get(1) - a.get(1)*b.get(0)
		])
	}

}

impl<T> Drop for Vector<T> {
	fn drop(&mut self) {
		let layout = std::alloc::Layout::array::<T>(self.dim).unwrap();
		unsafe {
			std::alloc::dealloc(self.ptr.as_ptr() as *mut u8, layout)
		}
	}
}

impl<T> std::fmt::Debug for Vector<T> where T: std::fmt::Debug {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		let mut components = "[".to_owned();
		for i in 0..self.dim-1 {
			components.push_str(&format!["{:?}, ", self.get(i)]);
		}
		components.push_str(&format!["{:?}]", self.get(self.dim-1)]);
		write!(f, "{components}")
	}
}

impl<T> PartialEq for Vector<T> where T: PartialEq {
	fn eq(&self, other: &Self) -> bool {
		if self.dim != other.dim {
			return false;
		}
		for i in 0..self.dim {
			if self.get(i) != other.get(i) {
				return false;
			}
		}
		true
	}
}

impl<T> Eq for Vector<T> where T: PartialEq {}

unsafe impl<T> Send for Vector<T> where T: Send {}

unsafe impl<T> Sync for Vector<T> where T: Sync {}

impl<T> Add for Vector<T> where T: Add<Output=T> {
	type Output = Self;
	fn add(self, other: Self) -> Self {
		if self.dim != other.dim {
			panic!("A sum of Vectors requires they be of the same dimension");
		}
		let mut sum = Vec::with_capacity(self.dim);
		for i in 0..self.dim {
			sum.push(self.get(i) + other.get(i));
		}
		sum.into()
	}
}

impl<T> Sub for Vector<T> where T: Sub<Output=T> {
	type Output = Self;
	fn sub(self, other: Self) -> Self {
		if self.dim != other.dim {
			panic!("A difference of Vectors requires they be of the same dimension");
		}
		let mut sum = Vec::with_capacity(self.dim);
		for i in 0..self.dim {
			sum.push(self.get(i) - other.get(i));
		}
		sum.into()
	}
}

impl<T> Mul for Vector<T> where T: Add<Output=T> + Mul<Output=T> {
	type Output = T;
	fn mul(self, other: Self) -> T {
		if self.dim != other.dim {
			panic!("A dot product of Vectors requires they be of the same dimension");
		}
		let mut total = self.get(0) * other.get(0);
		for i in 1..self.dim {
			total = total + self.get(i) * other.get(i);
		}
		total
	}
}

impl<T> Mul<T> for Vector<T> where T: Copy + Mul<Output=T> {
	type Output = Self;
	fn mul(self, other: T) -> Self {
		for i in 0..self.dim {
			unsafe { std::ptr::write(self.ptr.as_ptr().add(i), self.get(i) * other) }
		}
		self
	}
}

impl<T> Div<T> for Vector<T> where T: Copy + Div<Output=T> {
	type Output = Self;
	fn div(self, other: T) -> Self {
		for i in 0..self.dim {
			unsafe { std::ptr::write(self.ptr.as_ptr().add(i), self.get(i) / other) }
		}
		self
	}
}

impl<T> Neg for Vector<T> where T: Neg<Output=T> {
	type Output = Self;
	fn neg(self) -> Self {
		for i in 0..self.dim {
			unsafe { std::ptr::write(self.ptr.as_ptr().add(i), -self.get(i)) }
		}
		self
	}
}

impl<T> IntoIterator for Vector<T> {
	type Item = T;
	type IntoIter = VecIter<T>;
	fn into_iter(self) -> VecIter<T> {
		VecIter::new(self)
	}
}

impl<T, const N: usize> From<[T; N]> for Vector<T> {
	fn from(components: [T; N]) -> Self {
		Self::new(components)
	}
}

impl<T> From<Vec<T>> for Vector<T> {
	fn from(components: Vec<T>) -> Self {
		// set up the allocation
		let dim = components.len();
		let layout = std::alloc::Layout::array::<T>(dim).unwrap();
		let allocation = unsafe { std::alloc::alloc(layout) };
		let ptr = match std::ptr::NonNull::new(allocation as *mut T) {
			Some(p) => p,
			None => std::alloc::handle_alloc_error(layout)
		};
		// set the components
		unsafe {
			for (i, component) in components.into_iter().enumerate() {
				std::ptr::write(ptr.as_ptr().add(i), component);
			}
		}
		Self {
			ptr,
			dim
		}
	}
}

/// An iterator over a given [`Vector`].
pub struct VecIter<T> {
	vector: Vector<T>,
	index: usize
}

impl<T> VecIter<T> {

	/// Constructs a new `VecIter` from a [`Vector`].
	pub fn new(vector: Vector<T>) -> Self {
		Self {
			vector,
			index: 0
		}
	}

}

impl<T> Iterator for VecIter<T> {
	type Item = T;
	fn next(&mut self) -> Option<T> {
		if self.index >= self.vector.dim {
			return None;
		}
		let next_component = self.vector.get(self.index);
		self.index += 1;
		Some(next_component)
	}
}

#[cfg(test)]
mod vector {

	use super::{Vector, VecIter};

	mod dim {

		use super::Vector;

		#[test]
		fn zero() {
			assert_eq!(Vector::<usize>::new([]).dim, 0);
			assert_eq!(Vector::new([0]).dim, 1);
		}

		#[test]
		fn one() {
			assert_eq!(Vector::new([1]).dim, 1)
		}

		#[test]
		fn standard() {
			assert_eq!(Vector::new([1, 2, 3]).dim, 3)
		}

	}

	mod get {

		use super::Vector;

		#[test]
		fn standard() {
			let v = Vector::new([1, 2, 3, 4]);
			assert_eq!(v.get(0), 1);
			assert_eq!(v.get(1), 2);
			assert_eq!(v.get(2), 3);
			assert_eq!(v.get(3), 4)
		}

		#[test]
		#[should_panic]
		fn zero() {
			let v = Vector::new([]);
			let _: usize = v.get(0);
		}

		#[test]
		#[should_panic]
		fn overindexing() {
			let v = Vector::new([1, 2]);
			let _: usize = v.get(2);
		}

	}

	mod cross_product {

		use super::Vector;

		#[test]
		fn orthogonal() {
			let i_hat = Vector::new([1.0, 0.0, 0.0]);
			let j_hat = Vector::new([0.0, 1.0, 0.0]);
			assert_eq!(Vector::cross(i_hat, j_hat), Vector::new([0.0, 0.0, 1.0]))
		}

		#[test]
		fn anti_symmetry() {
			let left_i_hat = Vector::new([1.0, 0.0, 0.0]);
			let right_j_hat = Vector::new([0.0, 1.0, 0.0]);
			let left_j_hat = Vector::new([0.0, 1.0, 0.0]);
			let right_i_hat = Vector::new([1.0, 0.0, 0.0]);
			assert_eq!(Vector::cross(left_i_hat, right_j_hat), -Vector::cross(left_j_hat, right_i_hat))
		}

		#[test]
		fn parallel() {
			assert_eq!(
				Vector::cross(Vector::new([32.0, 2.0, 4.0]), Vector::new([32.0, 2.0, 4.0])),
				Vector::new([0.0, 0.0, 0.0])
			)
		}

	}

	mod equality {

		use super::Vector;

		mod zero {

			use super::Vector;

			#[test]
			fn no_args() {
				assert_ne!(Vector::<usize>::new([]), Vector::new([0, 0]))
			}

			#[test]
			fn with_args() {
				assert_ne!(Vector::new([0]), Vector::new([0, 0]))
			}

			#[test]
			fn between_no_args() {
				assert_eq!(Vector::<usize>::new([]), Vector::<usize>::new([]))
			}

		}

		#[test]
		fn differing_dimension() {
			assert_ne!(Vector::new([1, 2]), Vector::new([3]));
			assert_ne!(Vector::new([1, 2]), Vector::new([1]))
		}

		#[test]
		fn ne_same_dimension() {
			assert_ne!(Vector::new([1, 2]), Vector::new([3, 4]))
		}

		#[test]
		fn partially_eq_same_dimension() {
			assert_ne!(Vector::new([1, 2]), Vector::new([1, 4]));
			assert_ne!(Vector::new([1, 2]), Vector::new([3, 2]))
		}

		#[test]
		fn eq_same_dimension() {
			assert_eq!(Vector::new([1, 2, 3]), Vector::new([1, 2, 3]))
		}

	}

	mod addition {

		use super::Vector;

		mod zero {

			use super::Vector;

			#[test]
			#[should_panic]
			fn empty_zero() {
				let zero = Vector::new([]);
				let _ = Vector::new([1, 2, 3]) + zero;
			}

			#[test]
			fn standard() {
				let zero = Vector::new([0, 0, 0]);
				assert_eq!(Vector::new([1, 2, 3]) + zero, Vector::new([1, 2, 3]))
			}

		}

		#[test]
		#[should_panic]
		fn mismatched() {
			let _ = Vector::new([1, 2]) + Vector::new([1]);
		}

		#[test]
		fn associativity() {
			let a1 = Vector::new([1, 2]);
			let b1 = Vector::new([3, 4]);
			let c1 = Vector::new([5, 6]);
			let a2 = Vector::new([1, 2]);
			let b2 = Vector::new([3, 4]);
			let c2 = Vector::new([5, 6]);
			assert_eq!(a1 + (b1 + c1), (a2 + b2) + c2)
		}

		#[test]
		fn commutativity() {
			let left_v = Vector::new([1, 2, 3]);
			let right_w = Vector::new([4, 5, 6]);
			let left_w = Vector::new([4, 5, 6]);
			let right_v = Vector::new([1, 2, 3]);
			assert_eq!(left_v + right_w, left_w + right_v)
		}

		#[test]
		fn standard() {
			assert_eq!(
				Vector::new([1, 2]) + Vector::new([3, 4]), 
				Vector::new([4, 6])
			)
		}

		#[test]
		fn with_negative() {
			assert_eq!(
				Vector::new([1, 3]) + Vector::new([1, -2]),
				Vector::new([2, 1])
			)
		}

		#[test]
		fn both_negative() {
			assert_eq!(
				Vector::new([-1, -2]) + Vector::new([-3, -4]),
				Vector::new([-4, -6])
			)
		}

	}

	mod subtraction {

		use super::Vector;

		mod zero {

			use super::Vector;

			#[test]
			#[should_panic]
			fn empty_zero() {
				let zero = Vector::new([]);
				let _ = Vector::new([1, 2, 3]) - zero;
			}

			#[test]
			fn standard() {
				let zero = Vector::new([0, 0, 0]);
				assert_eq!(Vector::new([1, 2, 3]) - zero, Vector::new([1, 2, 3]))
			}

			#[test]
			fn backwards() {
				let zero = Vector::new([0, 0, 0]);
				assert_eq!(zero - Vector::new([1, 2, 3]), Vector::new([-1, -2, -3]))
			}


		}

		#[test]
		#[should_panic]
		fn mismatched() {
			let _ = Vector::new([1, 2]) - Vector::new([1]);
		}

		#[test]
		fn non_associativity() {
			let a1 = Vector::new([1, 2]);
			let b1 = Vector::new([3, 4]);
			let c1 = Vector::new([5, 6]);
			let a2 = Vector::new([1, 2]);
			let b2 = Vector::new([3, 4]);
			let c2 = Vector::new([5, 6]);
			assert_ne!(a1 - (b1 - c1), (a2 - b2) - c2)
		}

		#[test]
		fn non_commutativity() {
			let left_v = Vector::new([1, 2, 3]);
			let right_w = Vector::new([4, 5, 6]);
			let left_w = Vector::new([4, 5, 6]);
			let right_v = Vector::new([1, 2, 3]);
			assert_ne!(left_v - right_w, left_w - right_v)
		}

		#[test]
		fn standard() {
			assert_eq!(
				Vector::new([1, 2]) - Vector::new([3, 4]), 
				Vector::new([-2, -2])
			)
		}

		#[test]
		fn with_negative() {
			assert_eq!(
				Vector::new([1, 3]) - Vector::new([1, -2]),
				Vector::new([0, 5])
			)
		}

		#[test]
		fn both_negative() {
			assert_eq!(
				Vector::new([-1, -2]) - Vector::new([-3, -4]),
				Vector::new([2, 2])
			)
		}

	}

	mod dot_product {

		use super::Vector;

		mod zero {

			use super::Vector;

			#[test]
			#[should_panic]
			fn empty_zero() {
				let zero = Vector::new([]);
				let _ = Vector::new([1, 2, 3]) * zero;
			}

			#[test]
			#[should_panic]
			fn mismatched_zero() {
				let zero = Vector::new([0, 0]);
				let _ = Vector::new([1, 2, 3]) * zero;
			}

			#[test]
			fn standard() {
				let zero = Vector::new([0, 0, 0]);
				assert_eq!(Vector::new([1, 2, 3]) * zero, 0)
			}

		}

		#[test]
		#[should_panic]
		fn mismatched() {
			let _ = Vector::new([1, 2]) * Vector::new([1, 2, 3]);
		}

		#[test]
		fn commutativity() {
			let left_v = Vector::new([1, 2, 3]);
			let right_w = Vector::new([4, 5, 6]);
			let left_w = Vector::new([4, 5, 6]);
			let right_v = Vector::new([1, 2, 3]);
			assert_eq!(left_v * right_w, left_w * right_v)
		}

		#[test]
		fn standard() {
			assert_eq!(
				Vector::new([1, 2, 3]) * Vector::new([4, 5, 6]),
				32
			)
		}

		#[test]
		fn orthogonal() {
			assert_eq!(
				Vector::new([-4, 2]) * Vector::new([2, 4]),
				0
			)
		}

		#[test]
		fn parallel() {
			assert_eq!(
				Vector::new([0, 4]) * Vector::new([0, 3]),
				12
			)
		}

		#[test]
		fn opposite() {
			assert_eq!(
				Vector::new([1, 3]) * Vector::new([-2, -3]),
				-11
			)
		}

	}

	mod scalar_product {

		use super::Vector;

		mod zero {

			use super::Vector;

			#[test]
			fn empty_zero() {
				assert_eq!(Vector::new([]) * 3, Vector::new([]));
				assert_eq!(Vector::new([]) * 0, Vector::new([]))
			}

			#[test]
			fn mismatched_zero() {
				assert_eq!(Vector::new([0, 0]) * 34, Vector::new([0, 0]));
				assert_eq!(Vector::new([0, 0]) * 0, Vector::new([0, 0]))
			}

			#[test]
			fn standard() {
				assert_eq!(Vector::new([1, 2, 3]) * 0, Vector::new([0, 0, 0]))
			}

		}

		#[test]
		fn standard() {
			assert_eq!(Vector::new([1, 2, 3]) * 5, Vector::new([5, 10, 15]))
		}

		#[test]
		fn negative() {
			assert_eq!(Vector::new([1, 2, 3]) * -4, Vector::new([-4, -8, -12]))
		}

	}

	mod scalar_division {

		use super::Vector;

		mod zero {

			use super::Vector;

			#[test]
			fn empty_zero() {
				assert_eq!(Vector::new([]) / 3, Vector::new([]));
			}

			#[test]
			fn mismatched_zero() {
				assert_eq!(Vector::new([0, 0]) / 34, Vector::new([0, 0]));
			}

			#[test]
			#[should_panic]
			fn msimatched_zero_by_zero() {
				let _ = Vector::new([0, 0]) / 0;
			}

			#[test]
			#[should_panic]
			fn standard() {
				let _ = Vector::new([1, 2, 3]) / 0;
			}

		}

		#[test]
		fn standard() {
			assert_eq!(Vector::new([2, 4, 6]) / 2, Vector::new([1, 2, 3]))
		}

		#[test]
		fn negative() {
			assert_eq!(Vector::new([2, 4, 6]) / -2, Vector::new([-1, -2, -3]))
		}

	}

	mod negation {

		use super::Vector;

		mod zero {

			use super::Vector;

			#[test]
			fn empty_zero() {
				assert_eq!(-Vector::<isize>::new([]), Vector::<isize>::new([]));
			}

			#[test]
			fn mismatched_zero() {
				assert_eq!(-Vector::new([0, 0]), Vector::new([0, 0]));
			}
		}

		#[test]
		fn standard() {
			assert_eq!(-Vector::new([1, 2, 3]), Vector::new([-1, -2, -3]))
		}

		#[test]
		fn negative() {
			assert_eq!(-Vector::new([-1, -2, -3]), Vector::new([1, 2, 3]))
		}

	}

	mod iteration {

		use super::{Vector, VecIter};

		#[test]
		fn construction() {
			let _: VecIter<usize> = Vector::new([1, 2, 3]).into_iter();
		}

		#[test]
		fn last() {
			assert_eq!(Vector::new([1, 2]).into_iter().last().unwrap(), 2)
		}

	}

}

/// A container for the row and column counts of a [`Matrix`].
#[derive(Debug, PartialEq, Eq)]
pub struct MatrixDimensions {
	num_rows: usize,
	num_cols: usize
}

impl MatrixDimensions {

	/// Constructs a new `MatrixDimensions` from a row count (`num_rows`) and a
	/// a column count (`num_cols`).
	pub fn new(num_rows: usize, num_cols: usize) -> Self {
		Self { num_rows, num_cols }
	}

	/// Compares the given candidate dimensions to the dimensions encapsulated
	/// by `self`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::MatrixDimensions;
	/// let dims = MatrixDimensions::new(2, 3);
	/// assert!(dims.are(2, 3));
	/// ```
	pub fn are(&self, candidate_rows: usize, candidate_cols: usize) -> bool {
		self.num_rows == candidate_rows && self.num_cols == candidate_cols
	}

	/// Returns whether or not the matrix described by `self` is able to be
	/// right-multiplied by the matrix described by `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::MatrixDimensions;
	/// let a = MatrixDimensions::new(3, 2);
	/// let b = MatrixDimensions::new(2, 4);
	/// assert!(a.multiplies_with(&b));
	/// ```
	/// ```
	/// # use crate::space::linal::MatrixDimensions;
	/// let c = MatrixDimensions::new(2, 3);
	/// let d = MatrixDimensions::new(1, 3);
	/// assert!(!c.multiplies_with(&d));
	/// ```
	pub fn multiplies_with(&self, other: &Self) -> bool {
		self.num_cols == other.num_rows
	}

}

/// Matrix implementation for elements of type `T`.
#[derive(Debug)]
pub struct Matrix<T> {
	ptr: std::ptr::NonNull<T>,
	/// Encapsulates the dimensionality of the `Matrix`
	pub dims: MatrixDimensions
}

impl<T> Matrix<T> {

	/// Constructs a new `Matrix` from an array of component-containing rows.
	pub fn new<const R: usize, const C: usize>(rows: [[T; C]; R]) -> Self {
		// set up the allocation
		let layout = std::alloc::Layout::array::<T>(R*C).unwrap();
		let allocation = unsafe { std::alloc::alloc(layout) };
		let ptr = match std::ptr::NonNull::new(allocation as *mut T) {
			Some(p) => p,
			None => std::alloc::handle_alloc_error(layout)
		};
		// set the components
		for (i, row) in rows.into_iter().enumerate() {
			for (j, component) in row.into_iter().enumerate() {
				unsafe { std::ptr::write(ptr.as_ptr().add(C*i + j), component); }
			}
		}
		// set the dimensions
		let dims = if C == 0 {
			MatrixDimensions::new(0, 0)
		} else {
			MatrixDimensions::new(R, C)
		};
		Self { ptr, dims }
	}

	/// Constructs a new `Matrix` from a [`Vec`] of `Vec`s.
	pub fn with_vec(rows: Vec<Vec<T>>) -> Self {
		let num_rows = rows.len();
		let num_cols = rows[0].len();
		// check row regularity
		if !rows.iter().all(|r| r.len() == num_cols) {
			panic!("All rows of a Matrix must have the same number of columns")
		}
		// set up the allocation
		let layout = std::alloc::Layout::array::<T>(num_rows*num_cols).unwrap();
		let allocation = unsafe { std::alloc::alloc(layout) };
		let ptr = match std::ptr::NonNull::new(allocation as *mut T) {
			Some(p) => p,
			None => std::alloc::handle_alloc_error(layout)
		};
		// set the components
		for (i, row) in rows.into_iter().enumerate() {
			for (j, component) in row.into_iter().enumerate() {
				unsafe { std::ptr::write(ptr.as_ptr().add(num_cols*i + j), component); }
			}
		}
		// set the dimensions
		let dims = if num_cols == 0 {
			MatrixDimensions::new(0, 0)
		} else {
			MatrixDimensions::new(num_rows, num_cols)
		};
		Self { ptr, dims }
	}

	/// Returns the component of `self` specified by `candidate_row` and
	/// `candidate_col`.
	///
	/// # Panics
	/// If `self` is a zero-dimensional Matrix, a panic is issued.
	/// ```should_panic
	/// # use crate::space::linal::Matrix;
	/// let null = Matrix::<usize>::new::<0, 0>([]);
	/// let _: usize = null.get(0, 0); // `null` cannot be indexed!
	/// ```
	/// If `candidate_row` overindexes `self`, a similar panic is issued on
	/// the basis of dimensionality.
	/// ```should_panic
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 2], [3, 4]]);
	/// let _: usize = m.get(2, 0); // `m` only has two rows!
	/// ```
	/// In the same vein, if `candidate_col` overindexes `self`, a panic
	/// is again issued.
	/// ```should_panic
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 2], [3, 4]]);
	/// let _: usize = m.get(0, 2); // `m` only has two columns!
	/// ```
	///
	/// # Examples
	/// When provided with proper inputs, however, `get` is a powerful tool for
	/// analyzing the components of a `Matrix`.
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 2, 3], [4, 5, 6]]);
	/// assert_eq!(m.get(1, 2), 6);
	/// ```
	pub fn get(&self, candidate_row: usize, candidate_col: usize) -> T {
		if self.dims.are(0, 0) {
			panic!("Cannot index into a zero-dimensional Matrix")
		}
		if candidate_row >= self.dims.num_rows {
			panic!("Cannot index Matrix rows beyond {}", self.dims.num_rows)
		}
		if candidate_col >= self.dims.num_cols {
			panic!("Cannot index Matrix columns beyond {}", self.dims.num_cols)
		}
		unsafe { std::ptr::read(self.ptr.as_ptr().add(self.dims.num_cols*candidate_row + candidate_col)) }
	}

	/// Determines whether or not `self` is a square matrix.
	///
	/// This is checked simply by comparing the number of rows and columns in
	/// `self`. If `self` is a zero-by-zero matrix, this determines `self` to 
	/// be non-square.
	///	
	/// # Examples
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 2], [3, 4]]);
	/// assert!(m.is_square());
	/// ```
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let n = Matrix::new([[1, 2, 3], [4, 5, 6]]);
	/// assert!(!n.is_square());
	/// ```
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let null = Matrix::<usize>::new::<0, 0>([]);
	/// assert!(!null.is_square());
	/// ```
	pub fn is_square(&self) -> bool {
		if self.dims.are(0, 0) {
			return false;
		}
		self.dims.num_rows == self.dims.num_cols
	}


	/// Determines whether or not `self` is a row matrix.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let row = Matrix::new([[1, 2, 3]]);
	/// assert!(row.is_row());
	/// ```
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let col = Matrix::new([[1], [2]]);
	/// assert!(!col.is_row());
	/// ```
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let square = Matrix::new([[1, 2], [3, 4]]);
	/// assert!(!square.is_row());
	/// ```
	pub fn is_row(&self) -> bool {
		self.dims.num_rows == 1
	}

	/// Determines whether or not `self` is a column matrix.
	///
	/// While column matrices are in effect vectors, they are
	/// not interchangeable with [`Vector`] objects. The use
	/// of column matrices is discouraged primarily 
	/// 
	/// # Examples
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let col = Matrix::new([[1], [2]]);
	/// assert!(col.is_col());
	/// ```
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let row = Matrix::new([[1, 2, 3]]);
	/// assert!(!row.is_col());
	/// ```
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let square = Matrix::new([[1, 2], [3, 4]]);
	/// assert!(!square.is_col());
	/// ```
	pub fn is_col(&self) -> bool {
		self.dims.num_cols == 1
	}

	/// Returns whether or not `self` has the same dimensionality as `other`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let a = Matrix::new([[0, 1], [1, 0]]);
	/// let b = Matrix::new([[1, 0], [0, 1]]);
	/// assert!(a.summable_with(&b));
	/// ```
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let c = Matrix::new([[1, 2]]);
	/// let d = Matrix::new([[1], [2]]);
	/// assert!(!c.summable_with(&d));
	/// ```
	pub fn summable_with(&self, other: &Self) -> bool {
		self.dims == other.dims
	}

	/// Returns an iterator over the rows of `self`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 2], [3, 4]]);
	/// for (i, row) in m.rows().enumerate() {
	/// 	assert_eq!(row[0], 2*i + 1);
	/// 	assert_eq!(row[1], 2*i + 2);
	/// }
	/// ```
	pub fn rows<'a>(&'a self) -> MatrixRows<'a, T> {
		MatrixRows::new(&self.ptr, self.dims.num_rows, self.dims.num_cols)
	}

	/// Returns an iterator over the columns of `self`.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 2], [3, 4]]);
	/// for (i, col) in m.cols().enumerate() {
	/// 	assert_eq!(col[0], i + 1);
	/// 	assert_eq!(col[1], i + 3);
	/// }
	/// ```
	pub fn cols<'b>(&'b self) -> MatrixCols<'b, T> {
		MatrixCols::new(&self.ptr, self.dims.num_cols, self.dims.num_rows)
	}

	/// Returns the transpose of `self`.
	///
	/// This is just the standard transpose, both for square and non-square
	/// matrices.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 2], [3, 4]]);
	/// let transposed_m = Matrix::new([[1, 3], [2, 4]]);
	/// assert_eq!(m.transpose(), transposed_m);
	/// ```
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 2, 3], [4, 5, 6]]);
	/// let transposed_m = Matrix::new([[1, 4], [2, 5], [3, 6]]);
	/// assert_eq!(m.transpose(), transposed_m);
	/// ```
	pub fn transpose(&self) -> Self {
		let mut transpose_rows: Vec<Vec<T>> = Vec::with_capacity(self.dims.num_cols);
		for j in 0..self.dims.num_cols {
			let mut transpose_row: Vec<T> = Vec::with_capacity(self.dims.num_rows);
			for i in 0..self.dims.num_rows {
				transpose_row.push(self.get(i, j));
			}
			transpose_rows.push(transpose_row);
		}
		Self::with_vec(transpose_rows)
	}

}

impl<T> Matrix<T> where T: Default + PartialEq {

	/// Determines whether or not `self` is a diagonal matrix.
	///
	/// The first property of note is that diagonal matrices must be square.
	/// If `self` is non-square, it is not diagonal.
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 2, 3], [4, 5, 6]]);
	/// assert!(!m.is_diagonal());
	/// ```
	/// The defining property of diagonal matrices, however, is that they are
	/// only non-zero on their diagonal. This requires a distinction between
	/// zero and non-zero `T` values. The distinction here is made through
	/// the `default` value of `T`, which is guaranteed through the [`Default`]
	/// bound for this method.
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 0], [0, 3]]);
	/// assert!(m.is_diagonal());
	/// ```
	/// This could potentially have unintended consequences, particularly for
	/// non-primitive types.
	/// ```
	/// # use crate::space::linal::Matrix;
	/// #[derive(PartialEq)]
	/// struct Wrapper {
	///		inner: u8
	/// }
	///
	/// impl Wrapper {
	///		pub fn new(inner: u8) -> Self {
	///			 Self { inner }
	///		}
	/// }
	/// 
	/// impl Default for Wrapper {
	///		fn default() -> Self {
	///			Self::new(1)
	/// 	}
	/// }
	///
	/// let attempted_identity = Matrix::new([
	/// 	[Wrapper::new(1), Wrapper::new(0)],
	///		[Wrapper::new(0), Wrapper::new(1)]
	/// ]);
	///
	/// assert!(!attempted_identity.is_diagonal());
	///
	/// let quirky_identity = Matrix::new([
	/// 	[Wrapper::new(0), Wrapper::new(1)],
	/// 	[Wrapper::new(1), Wrapper::new(0)]
	/// ]);
	///
	/// assert!(quirky_identity.is_diagonal());
	/// ```
	pub fn is_diagonal(&self) -> bool {
		if !self.is_square() {
			return false;
		}
		for i in 0..self.dims.num_rows {
			for j in 0..self.dims.num_cols {
				if i != j && self.get(i, j) != T::default() {
					return false;
				}
			}
		}
		true
	}

	/// Determines the triangularity properties of `self`.
	///
	/// If `self` is upper-triangular, lower-triangular, or just diagonal, then
	/// a `Some` value is returned containing a [`Triangularity`] designation.
	/// ```
	/// # use crate::space::linal::{Matrix, Triangularity};
	/// let m = Matrix::new([
	/// 	[1, 0, 0],
	/// 	[2, 3, 0],
	/// 	[5, 3, 1]
	/// ]);
	/// let triangularity: Triangularity = m.triangularity().unwrap();
	/// assert!(matches![triangularity, Triangularity::LowerTriangular]);
	/// ```
	/// ```
	/// # use crate::space::linal::{Matrix, Triangularity};
	/// let m = Matrix::new([
	/// 	[1, 2, 3],
	/// 	[0, 3, 4],
	/// 	[0, 0, 1]
	/// ]);
	/// let triangularity: Triangularity = m.triangularity().unwrap();
	/// assert!(matches![triangularity, Triangularity::UpperTriangular]);
	/// ```
	/// ```
	/// # use crate::space::linal::{Matrix, Triangularity};
	/// let m = Matrix::new([
	/// 	[1, 0, 0],
	/// 	[0, 3, 0],
	/// 	[0, 0, 4]
	/// ]);
	/// let triangularity: Triangularity = m.triangularity().unwrap();
	/// assert!(matches![triangularity, Triangularity::Diagonal]);
	/// ```
	/// If `self` does not have a triangularity designation (ie. it is not
	/// uppper-triangular, lower-triangular, or diagonal), a `None` is returned
	/// to indicate there is no `Triangularity` value matching its type.
	/// ```
	/// # use crate::space::linal::{Matrix, Triangularity};
	/// let m = Matrix::new([
	/// 	[1, 2, 3],
	/// 	[0, 3, 4],
	/// 	[4, 0, 1]
	/// ]);
	/// assert!(m.triangularity().is_none());
	/// ```
	/// There are two other important things to note here. The first is a
	/// further detail of the mathematical notion of matrix triangularity,
	/// which requires that matrices must be square to be in any state of
	/// triangularity. Accordingly, non-square values of `self` recieve a
	/// `None` classification.
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([
	/// 	[1, 2, 3, 4],
	/// 	[0, 3, 4, 2],
	/// 	[0, 0, 1, 8]
	/// ]);
	/// assert!(m.triangularity().is_none());
	/// ```
	/// The last item of note is on the implementation of this method.
	/// Because triangularity at its heart rests on a distinction between zero
	/// and non-zero values and `T` is a generic type, `T::default()` is used
	/// to be the definitive notion of zero for `T`. This can have unintended
	/// consequences.
	/// ```
	/// # use crate::space::linal::{Matrix, Triangularity};
	/// #[derive(PartialEq)]
	/// struct Wrapper {
	///		inner: u8
	/// }
	///
	/// impl Wrapper {
	///		pub fn new(inner: u8) -> Self {
	///			 Self { inner }
	///		}
	/// }
	///
	/// impl Default for Wrapper {
	///		fn default() -> Self {
	///			Self::new(1)
	/// 	}
	/// }
	///
	/// let attempted_diagonal = Matrix::new([
	/// 	[Wrapper::new(1), Wrapper::new(0)],
	///		[Wrapper::new(0), Wrapper::new(4)]
	/// ]);
	///
	/// assert!(attempted_diagonal.triangularity().is_none());
	///
	/// let quirky_diagonal = Matrix::new([
	/// 	[Wrapper::new(4), Wrapper::new(1)],
	/// 	[Wrapper::new(1), Wrapper::new(2)]
	/// ]);
	///
	/// let quirky_triangularity = quirky_diagonal.triangularity().unwrap();
	///
	/// assert!(matches![quirky_triangularity, Triangularity::Diagonal]);
	/// ```
	pub fn triangularity(&self) -> Option<Triangularity> {
		if !self.is_square() {
			return None;
		}
		let mut is_upper_triangular = true;
		let mut is_lower_triangular = true;
		for i in 0..self.dims.num_rows {
			for j in 0..self.dims.num_cols {
				if j < i && self.get(i, j) != T::default() {
					is_upper_triangular = false;
				}
				if j > i && self.get(i, j) != T::default() {
					is_lower_triangular = false;
				}
			}
		}
		if is_upper_triangular && !is_lower_triangular {
			return Some(Triangularity::UpperTriangular);
		}
		if is_lower_triangular && !is_upper_triangular {
			return Some(Triangularity::LowerTriangular);
		}
		if is_upper_triangular && is_lower_triangular {
			return Some(Triangularity::Diagonal);
		}
		None
	}

}

impl<T> Matrix<T> where T: Add<Output=T> {

	/// Computes the trace of `self`.
	///
	/// # Panics
	/// This is only defined for square matrices. If `self` is null or
	/// otherwise non-square, a panic is issued.
	/// ```should_panic
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::<usize>::new::<0, 0>([]);
	/// let _ = m.trace();
	/// ```
	/// ```should_panic
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 2, 3], [4, 5, 6]]);
	/// let _ = m.trace();
	/// ```
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 2], [3, 4]]);
	/// assert_eq!(m.trace(), 5);
	/// ```
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1]]);
	/// assert_eq!(m.trace(), 1);
	/// ```
	pub fn trace(&self) -> T {
		if !self.is_square() {
			panic!("Cannot take the trace of a non-square matrix")
		}
		let mut sum: T = self.get(0, 0);
		for i in 1..self.dims.num_rows {
			sum = sum + self.get(i, i);
		}
		sum
	}

}

impl<T> Matrix<T> where T: Default + PartialEq + Add<Output=T> {

	/// Determines the dimensionality of the null space of `self`.
	///
	/// If the sum of the values of any row is zero, this adds to the
	/// dimensionality of the null space of `self`. 
	///
	/// # Warning
	/// Null space not located in the linear subspace spanned by the 
	/// basis vectors of `self` is not yet accounted for in the algorithm
	/// utilized here.
	///
	/// # Examples
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 0], [1, -1]]);
	/// assert_eq!(m.null_dim(), 1);
	/// ```
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 1, 3], [-1, -1, 2]]);
	/// assert_eq!(m.null_dim(), 2);
	/// ```
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 1], [1, -1], [4, 3]]);
	/// assert_eq!(m.null_dim(), 1);
	/// ```
	pub fn null_dim(&self) -> usize {
		let mut null_dimensionality = if self.dims.num_cols > self.dims.num_rows {
			self.dims.num_cols - self.dims.num_rows
		} else {
			0
		};
		for row in self.rows() {
			let mut row_sum = T::default();
			for element in row.into_iter() {
				row_sum = row_sum + element;
			}
			if row_sum == T::default() {
				null_dimensionality += 1;
			}
		}
		null_dimensionality
	}

	/// Determines the rank of `self`.
	///
	/// The output dimensionality of `self` is exactly equivalent to the
	/// difference of the number of columns in `self` and the dimensionality
	/// of its null space.
	/// 
	/// # Examples
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 0], [1, -1]]);
	/// assert_eq!(m.rank(), 1);
	/// ```
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 1, 3], [-1, -1, 2]]);
	/// assert_eq!(m.rank(), 1);
	/// ```
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([[1, 1], [1, -1], [4, 3]]);
	/// assert_eq!(m.rank(), 2);
	/// ```
	pub fn rank(&self) -> usize {
		if self.dims.num_rows > self.dims.num_cols {
			return self.dims.num_rows - self.null_dim();
		}
		self.dims.num_cols - self.null_dim()
	}

}

impl<T> Matrix<T> where T: Default + PartialEq + Neg<Output=T> {

	/// Determines the symmetry properties of `self`.
	///
	/// If `self` is symmetric or skew-symmetric, then a `Some` value is
	/// returned containing a [`Symmetry`] designation.
	/// ```
	/// # use crate::space::linal::{Matrix, Symmetry};
	/// let m = Matrix::new([
	/// 	[1, 2, 3],
	/// 	[2, 3, 4],
	/// 	[3, 4, 1]
	/// ]);
	/// assert!(matches![m.symmetry().unwrap(), Symmetry::Symmetric]);
	/// ```
	/// ```
	/// # use crate::space::linal::{Matrix, Symmetry};
	/// let m = Matrix::new([
	/// 	[0, 2, 3],
	/// 	[-2, 0, 4],
	/// 	[-3, -4, 0]
	/// ]);
	/// assert!(matches![m.symmetry().unwrap(), Symmetry::Skew]);
	/// ```
	/// If `self` is neither symmetric or skew-symmetric, a `None` value is
	/// emitted.
	/// ```
	/// # use crate::space::linal::{Matrix, Symmetry};
	/// let m = Matrix::new([
	/// 	[1, 2, 3],
	/// 	[2, 0, 4],
	/// 	[4, 4, 1]
	/// ]);
	/// assert!(m.symmetry().is_none());
	/// ```
	/// The last item of note is a further detail of the mathematical notion
	/// of matrix symmetry, which requires that matrices must be square to be
	/// in any state of symmetry. Accordingly, non-square values of `self` also
	/// emit a `None`.
	/// ```
	/// # use crate::space::linal::Matrix;
	/// let m = Matrix::new([
	/// 	[1, 2, 0, 4],
	/// 	[2, 1, 1, 2],
	/// 	[4, 0, 2, 1]
	/// ]);
	/// assert!(m.symmetry().is_none());
	/// ```
	pub fn symmetry(&self) -> Option<Symmetry> {
		if !self.is_square() {
			return None;
		}
		let mut is_symmetric = true;
		let mut is_skew = true;
		let transpose = self.transpose();
		for i in 0..self.dims.num_rows {
			for j in 0..self.dims.num_cols {
				if self.get(i, j) != transpose.get(i, j) {
					is_symmetric = false;
				}
				if self.get(i, j) != -transpose.get(i, j) {
					is_skew = false;
				}
			}
		}
		if is_symmetric {
			return Some(Symmetry::Symmetric);
		}
		if is_skew {
			return Some(Symmetry::Skew);
		}
		None
	}

}

unsafe impl<T> Send for Matrix<T> where T: Send {}

unsafe impl<T> Sync for Matrix<T> where T: Sync {}

impl<T> PartialEq for Matrix<T> where T: PartialEq {
	fn eq(&self, other: &Self) -> bool {
		if self.dims != other.dims {
			return false;
		}
		for i in 0..self.dims.num_rows {
			for j in 0..self.dims.num_cols {
				if self.get(i, j) != other.get(i, j) {
					return false;
				}
			}
		}
		true
	}
}

impl<T> Add for Matrix<T> where T: Add<Output=T> {
	type Output = Self;
	fn add(self, other: Self) -> Self {
		if !self.summable_with(&other) {
			panic!("Addition can only occur between matrices of the same dimension")
		}
		let mut sum_components = Vec::with_capacity(self.dims.num_rows);
		for i in 0..self.dims.num_rows {
			let mut row = Vec::with_capacity(self.dims.num_cols);
			for j in 0..self.dims.num_cols {
				row.push(self.get(i, j) + other.get(i, j));
			}
			sum_components.push(row);
		}
		Self::with_vec(sum_components)
	}
}

impl<T> Sub for Matrix<T> where T: Sub<Output=T> {
	type Output = Self;
	fn sub(self, other: Self) -> Self {
		if !self.summable_with(&other) {
			panic!("Subtraction can only occur between matrices of the same dimension")
		}
		let mut difference_components = Vec::with_capacity(self.dims.num_rows);
		for i in 0..self.dims.num_rows {
			let mut row = Vec::with_capacity(self.dims.num_cols);
			for j in 0..self.dims.num_cols {
				row.push(self.get(i, j) - other.get(i, j));
			}
			difference_components.push(row);
		}
		Self::with_vec(difference_components)
	}
}

/// Computes `Matrix`-`Matrix` products.
impl<T> Mul for Matrix<T> where T: Copy + Sub<Output=T> + Add<Output=T> + Mul<Output=T> {
	type Output = Self;
	fn mul(self, other: Self) -> Self {
		if !self.dims.multiplies_with(&other.dims) {
			panic!("Matrices do not have compatible dimensions for multiplication")
		}
		let zero = self.get(0, 0) - self.get(0, 0);
		let mut product = vec![vec![zero; self.dims.num_rows]; other.dims.num_cols];
		for i in 0..self.dims.num_rows {
			for j in 0..other.dims.num_cols {
				let mut sum = zero;
				for k in 0..self.dims.num_cols {
					sum = sum + self.get(i, k)*other.get(k, j);
				}
				product[i][j] = sum;
			}
		}
		Self::with_vec(product)
	}
}

/// Computes `Matrix`-[`Vector`] products.
impl<T> Mul<Vector<T>> for Matrix<T> where T: Copy + Sub<Output=T> + Add<Output=T> + Mul<Output=T> {
	type Output = Self;
	fn mul(self, other: Vector<T>) -> Self {
		if self.dims.num_cols != other.dim {
			panic!("Vector of dimension {} is not compatible with a Matrix of dimension {}x{}", other.dim, self.dims.num_rows, self.dims.num_cols);
		}
		let zero = self.get(0, 0) - self.get(0, 0);
		let mut product = vec![zero; self.dims.num_rows];
		for i in 0..self.dims.num_rows {
			let mut sum = zero;
			for j in 0..self.dims.num_cols {
				sum = sum + self.get(i, j)*other.get(j);
			}
			product[i] = sum;
		}
		Self::with_vec(vec![product])
	}
}

/// Computes scalar-`Matrix` products.
impl<T> Mul<T> for Matrix<T> where T: Copy + Mul<Output=T> {
	type Output = Self;
	fn mul(self, other: T) -> Self {
		let mut difference_components = Vec::with_capacity(self.dims.num_rows);
		for i in 0..self.dims.num_rows {
			let mut row = Vec::with_capacity(self.dims.num_cols);
			for j in 0..self.dims.num_cols {
				row.push(self.get(i, j) * other);
			}
			difference_components.push(row);
		}
		Self::with_vec(difference_components)
	}
}

impl<T> Div<T> for Matrix<T> where T: Copy + Div<Output=T> {
	type Output = Self;
	fn div(self, other: T) -> Self {
		let mut difference_components = Vec::with_capacity(self.dims.num_rows);
		for i in 0..self.dims.num_rows {
			let mut row = Vec::with_capacity(self.dims.num_cols);
			for j in 0..self.dims.num_cols {
				row.push(self.get(i, j) / other);
			}
			difference_components.push(row);
		}
		Self::with_vec(difference_components)
	}
}

impl<T> Neg for Matrix<T> where T: Neg<Output=T> {
	type Output = Self;
	fn neg(self) -> Self {
		let mut difference_components = Vec::with_capacity(self.dims.num_rows);
		for i in 0..self.dims.num_rows {
			let mut row = Vec::with_capacity(self.dims.num_cols);
			for j in 0..self.dims.num_cols {
				row.push(-self.get(i, j));
			}
			difference_components.push(row);
		}
		Self::with_vec(difference_components)
	}
}

impl<T> Eq for Matrix<T> where T: PartialEq {}

/// Markers for the triangularity of a [`Matrix`].
///
/// Given a square matrix, there are three triangularity properties it can
/// exhibit (assuming it exhibits any at all). If it is `Diagonal`, it is
/// non-zero only on its primary diagonal; if it is `UpperTriangular`, then
/// it is non-zero only on the entries _above_ and including the primary
/// diagonal; if it is `LowerTriangular`, it is non-zero only on the entries
/// _below_ and including the primary diagonal.
pub enum Triangularity {
	/// Marks a diagonal `Matrix`.
	Diagonal,
	/// Marks an upper-triangular `Matrix`.
	UpperTriangular,
	/// Marks a lower-triangular `Matrix`.
	LowerTriangular
}

/// Markers for the symmetry of a [`Matrix`].
///
/// Given a square matrix, there are two symmetry properties it can exhibit
/// (assuming it exhibits any at all). If some matrix _A_ is symmetric (ie.
/// `Symmetric`), we have that _A_ = _A_<sup>T</sup>. If _A_ = -_A_<sup>T</sup>,
/// then _A_ is said to be skew-symmetric (ie. `Skew`).
pub enum Symmetry {
	/// Marks a symmetric `Matrix`.
	Symmetric,
	/// Marks a skew-symmetric `Matrix`.
	Skew
}

/// An iterator over a given [`Matrix`]'s rows.
pub struct MatrixRows<'a, T> {
	ptr: &'a std::ptr::NonNull<T>,
	num_rows: usize,
	elements_per_row: usize,
	index: usize
}

impl<'a, T> MatrixRows<'a, T> {

	/// Constructs a new `MatrixRows` from relevant `Matrix` information.
	pub fn new(ptr: &'a std::ptr::NonNull<T>, num_rows: usize, elements_per_row: usize) -> Self {
		Self {
			ptr,
			num_rows,
			elements_per_row,
			index: 0
		}
	}

}

impl<'a, T> Iterator for MatrixRows<'a, T> {
	type Item = Vec<T>;
	fn next(&mut self) -> Option<Vec<T>> {
		if self.index >= self.num_rows {
			return None;
		}
		let mut next_row = Vec::with_capacity(self.elements_per_row);
		for i in 0..self.elements_per_row {
			let element = unsafe { 
				std::ptr::read(self.ptr.as_ptr().add(self.index*self.elements_per_row + i))
			};
			next_row.push(element);
		}
		self.index += 1;
		Some(next_row)
	}
}

/// An iterator over a given [`Matrix`]'s columns.
pub struct MatrixCols<'a, T> {
	ptr: &'a std::ptr::NonNull<T>,
	num_cols: usize,
	elements_per_col: usize,
	index: usize
}


impl<'a, T> MatrixCols<'a, T> {

	/// Constructs a new `MatrixCols` from relevant `Matrix` information.
	pub fn new(ptr: &'a std::ptr::NonNull<T>, num_cols: usize, elements_per_col: usize) -> Self {
		Self {
			ptr,
			num_cols,
			elements_per_col,
			index: 0
		}
	}

}

impl<'a, T> Iterator for MatrixCols<'a, T> {
	type Item = Vec<T>;
	fn next(&mut self) -> Option<Vec<T>> {
		if self.index >= self.num_cols {
			return None;
		}
		let mut next_col = Vec::with_capacity(self.elements_per_col);
		for i in 0..self.elements_per_col {
			let element = unsafe {
				std::ptr::read(self.ptr.as_ptr().add(self.index + self.elements_per_col*i))
			};
			next_col.push(element);
		}
		self.index += 1;
		Some(next_col)
	}
}

#[cfg(test)]
mod matrix {

	use super::Matrix;

	mod dims {

		use super::Matrix;

		mod null {

			use super::Matrix;

			#[test]
			fn null_matrix() {
				let m = Matrix::<usize>::new::<0, 0>([]);
				assert!(m.dims.are(0, 0))
			}

			#[test]
			fn null_rows() {
				let m = Matrix::<usize>::new([[]]);
				assert!(m.dims.are(0, 0))
			}

			#[test]
			fn null_cols() {
				let m = Matrix::<usize>::new([[], []]);
				assert!(m.dims.are(0, 0))
			}


		}

		mod zero {

			use super::Matrix;

			#[test]
			fn square() {
				let m = Matrix::new([
					[0, 0],
					[0, 0]
				]);
				assert!(m.dims.are(2, 2))
			}

			#[test]
			fn row() {
				let m = Matrix::new([[0, 0]]);
				assert!(m.dims.are(1, 2))
			}

			#[test]
			fn column() {
				let m = Matrix::new([[0], [0]]);
				assert!(m.dims.are(2, 1))
			}

		}

		#[test]
		fn standard() {
			let m = Matrix::new([
				[1, 2, 3],
				[4, 5, 6],
				[7, 8, 9],
				[10, 11, 12]
			]);
			assert!(m.dims.are(4, 3))
		}



	}

	mod get {

		use super::Matrix;

		mod null {

			use super::Matrix;

			#[test]
			#[should_panic]
			fn no_rows() {
				let m = Matrix::<usize>::new::<0, 0>([]);
				let _ = m.get(0, 0);
			}

			#[test]
			#[should_panic]
			fn no_cols() {
				let m = Matrix::<usize>::new([[]]);
				let _ = m.get(0, 0);
			}

		}

		#[test]
		fn square() {
			let m = Matrix::new([
				[1, 2],
				[3, 4]
			]);
			assert_eq!(m.get(0, 0), 1);
			assert_eq!(m.get(0, 1), 2);
			assert_eq!(m.get(1, 0), 3);
			assert_eq!(m.get(1, 1), 4)
		}

		#[test]
		fn right_leaning() {
			let m = Matrix::new([
				[1, 2, 3],
				[4, 5, 6]
			]);
			assert_eq!(m.get(0, 0), 1);
			assert_eq!(m.get(0, 1), 2);
			assert_eq!(m.get(0, 2), 3);
			assert_eq!(m.get(1, 0), 4);
			assert_eq!(m.get(1, 1), 5);
			assert_eq!(m.get(1, 2), 6)
		}

		#[test]
		fn top_heavy() {
			let m = Matrix::new([
				[1, 2],
				[3, 4],
				[5, 6]
			]);
			assert_eq!(m.get(0, 0), 1);
			assert_eq!(m.get(0, 1), 2);
			assert_eq!(m.get(1, 0), 3);
			assert_eq!(m.get(1, 1), 4);
			assert_eq!(m.get(2, 0), 5);
			assert_eq!(m.get(2, 1), 6)
		}

		#[test]
		#[should_panic]
		fn overindex_row() {
			let m = Matrix::new([
				[1, 2],
				[3, 4]
			]);
			let _ = m.get(2, 0);
		}

		#[test]
		#[should_panic]
		fn overindex_column() {
			let m = Matrix::new([
				[1, 2],
				[3, 4]
			]);
			let _ = m.get(0, 2);
		}

	}

	mod addition {

		use super::Matrix;

		mod zero {

			use super::Matrix;

			mod square {

				use super::Matrix;

				#[test]
				fn left_identity() {
					let zero = Matrix::new([[0, 0], [0, 0]]);
					let m = Matrix::new([[1, 2], [3, 4]]);
					let expected = Matrix::new([[1, 2], [3, 4]]);
					assert_eq!(zero + m, expected)
				}

				#[test]
				fn right_identity() {
					let zero = Matrix::new([[0, 0], [0, 0]]);
					let m = Matrix::new([[1, 2], [3, 4]]);
					let expected = Matrix::new([[1, 2], [3, 4]]);
					assert_eq!(m + zero, expected)
				}

			}

			mod horizontal_rectangular {

				use super::Matrix;

				#[test]
				fn left_identity() {
					let zero = Matrix::new([[0, 0, 0], [0, 0, 0]]);
					let m = Matrix::new([[1, 2, 3], [4, 5, 6]]);
					let expected = Matrix::new([[1, 2, 3], [4, 5, 6]]);
					assert_eq!(zero + m, expected)
				}

				#[test]
				fn right_identity() {
					let zero = Matrix::new([[0, 0, 0], [0, 0, 0]]);
					let m = Matrix::new([[1, 2, 3], [4, 5, 6]]);
					let expected = Matrix::new([[1, 2, 3], [4, 5, 6]]);
					assert_eq!(m + zero, expected)
				}

				

			}

			mod vertical_rectangular {

				use super::Matrix;

				#[test]
				fn left_identity() {
					let zero = Matrix::new([[0, 0], [0, 0], [0, 0]]);
					let m = Matrix::new([[1, 2], [3, 4], [5, 6]]);
					let expected = Matrix::new([[1, 2], [3, 4], [5, 6]]);
					assert_eq!(zero + m, expected)
				}

				#[test]
				fn right_identity() {
					let zero = Matrix::new([[0, 0], [0, 0], [0, 0]]);
					let m = Matrix::new([[1, 2], [3, 4], [5, 6]]);
					let expected = Matrix::new([[1, 2], [3, 4], [5, 6]]);
					assert_eq!(m + zero, expected)
				}


			}

		}

		#[test]
		fn square() {
			let a = Matrix::new([[1, 2], [3, 4]]);
			let b = Matrix::new([[5, 6], [7, 8]]);
			let sum = Matrix::new([[6, 8], [10, 12]]);
			assert_eq!(a + b, sum)
		}

		#[test]
		fn horizontal_rectangular() {
			let a = Matrix::new([[1, 2, 3], [4, 5, 6]]);
			let b = Matrix::new([[7, 8, 9], [10, 11, 12]]);
			let sum = Matrix::new([[8, 10, 12], [14, 16, 18]]);
			assert_eq!(a + b, sum)
		}

		#[test]
		fn vertical_rectangular() {
			let a = Matrix::new([[1, 2], [3, 4], [5, 6]]);
			let b = Matrix::new([[7, 8], [9, 10], [11, 12]]);
			let sum = Matrix::new([[8, 10], [12, 14], [16, 18]]);
			assert_eq!(a + b, sum)
		}

	}

	mod subtraction {

		use super::Matrix;

		mod zero {

			use super::Matrix;

			mod square {

				use super::Matrix;

				#[test]
				fn left_identity() {
					let zero = Matrix::new([[0, 0], [0, 0]]);
					let m = Matrix::new([[1, 2], [3, 4]]);
					let expected = Matrix::new([[-1, -2], [-3, -4]]);
					assert_eq!(zero - m, expected)
				}

				#[test]
				fn right_identity() {
					let zero = Matrix::new([[0, 0], [0, 0]]);
					let m = Matrix::new([[1, 2], [3, 4]]);
					let expected = Matrix::new([[1, 2], [3, 4]]);
					assert_eq!(m - zero, expected)
				}

			}

			mod horizontal_rectangular {

				use super::Matrix;

				#[test]
				fn left_identity() {
					let zero = Matrix::new([[0, 0, 0], [0, 0, 0]]);
					let m = Matrix::new([[1, 2, 3], [4, 5, 6]]);
					let expected = Matrix::new([[-1, -2, -3], [-4, -5, -6]]);
					assert_eq!(zero - m, expected)
				}

				#[test]
				fn right_identity() {
					let zero = Matrix::new([[0, 0, 0], [0, 0, 0]]);
					let m = Matrix::new([[1, 2, 3], [4, 5, 6]]);
					let expected = Matrix::new([[1, 2, 3], [4, 5, 6]]);
					assert_eq!(m - zero, expected)
				}

				

			}

			mod vertical_rectangular {

				use super::Matrix;

				#[test]
				fn left_identity() {
					let zero = Matrix::new([[0, 0], [0, 0], [0, 0]]);
					let m = Matrix::new([[1, 2], [3, 4], [5, 6]]);
					let expected = Matrix::new([[-1, -2], [-3, -4], [-5, -6]]);
					assert_eq!(zero - m, expected)
				}

				#[test]
				fn right_identity() {
					let zero = Matrix::new([[0, 0], [0, 0], [0, 0]]);
					let m = Matrix::new([[1, 2], [3, 4], [5, 6]]);
					let expected = Matrix::new([[1, 2], [3, 4], [5, 6]]);
					assert_eq!(m - zero, expected)
				}


			}

		}

		#[test]
		fn square() {
			let a = Matrix::new([[1, 2], [3, 4]]);
			let b = Matrix::new([[5, 6], [7, 8]]);
			let sum = Matrix::new([[-4, -4], [-4, -4]]);
			assert_eq!(a - b, sum)
		}

		#[test]
		fn horizontal_rectangular() {
			let a = Matrix::new([[1, 2, 3], [4, 5, 6]]);
			let b = Matrix::new([[7, 8, 9], [10, 11, 12]]);
			let sum = Matrix::new([[-6, -6, -6], [-6, -6, -6]]);
			assert_eq!(a - b, sum)
		}

		#[test]
		fn vertical_rectangular() {
			let a = Matrix::new([[1, 2], [3, 4], [5, 6]]);
			let b = Matrix::new([[7, 8], [9, 10], [11, 12]]);
			let sum = Matrix::new([[-6, -6], [-6, -6], [-6, -6]]);
			assert_eq!(a - b, sum)
		}

	}

	mod matrix_multiplication {

		use super::Matrix;

		mod zero {

			use super::Matrix;

			#[test]
			fn square() {
				let zero = Matrix::new([[0, 0], [0, 0]]);
				let m = Matrix::new([[1, 2], [3, 4]]);
				assert_eq!(zero * m, Matrix::new([[0, 0], [0, 0]]))
			}

			#[test]
			fn rectangular() {
				let zero = Matrix::new([[0, 0], [0, 0], [0, 0]]);
				let m = Matrix::new([[1, 2, 3], [4, 5, 6]]);
				assert_eq!(m * zero, Matrix::new([[0, 0], [0, 0]]))
			}

		}

		#[test]
		fn identity() {
			let identity = Matrix::new([[1, 0], [0, 1]]);
			let m = Matrix::new([[1, 2], [3, 4]]);
			assert_eq!(m * identity, Matrix::new([[1, 2], [3, 4]]));
		}

		#[test]
		fn square() {
			let a = Matrix::new([[1, 2], [3, 4]]);
			let b = Matrix::new([[5, 6], [7, 8]]);
			let product = Matrix::new([[19, 22], [43, 50]]);
			assert_eq!(a * b, product)
		}

		#[test]
		fn rectangular() {
			let a = Matrix::new([[1, 2], [3, 4], [5, 6]]);
			let b = Matrix::new([[1, 2, 3], [4, 5, 6]]);
			let product = Matrix::new([
				[9, 12, 15],
				[19, 26, 33],
				[29, 40, 51]
			]);
			assert_eq!(a * b, product)
		}

	}

	mod scalar_multiplication {

		use super::Matrix;

		mod zero_matrix {

			use super::Matrix;

			#[test]
			fn by_zero() {
				let zero = Matrix::new([[0, 0], [0, 0]]);
				let product = Matrix::new([[0, 0], [0, 0]]);
				assert_eq!(zero * 0, product)
			}

			#[test]
			fn by_arbitrary() {
				let zero = Matrix::new([[0, 0], [0, 0]]);
				let product = Matrix::new([[0, 0], [0, 0]]);
				assert_eq!(zero * 5, product)
			}

		}

		#[test]
		fn standard() {
			let m = Matrix::new([[1, 2], [3, 4]]);
			let product = Matrix::new([[2, 4], [6, 8]]);
			assert_eq!(m * 2, product)
		}

		#[test]
		fn horizontal_rectangular() {
			let m = Matrix::new([[1, 2, 3], [4, 5, 6]]);
			let product = Matrix::new([[2, 4, 6], [8, 10, 12]]);
			assert_eq!(m * 2, product)
		}

		#[test]
		fn vertical_rectangular() {
			let m = Matrix::new([[1, 2], [3, 4], [5, 6]]);
			let product = Matrix::new([[2, 4], [6, 8], [10, 12]]);
			assert_eq!(m * 2, product)
		}

	}

}