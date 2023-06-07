//! Linear algebra support.

use std::ops::{Add, Sub, Mul, Div};

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

#[cfg(test)]
mod vector {

	use super::Vector;

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

}