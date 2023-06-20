//! Useful traits for numeric types.

/// The unary square root operation.
pub trait SquareRoot {
	type Output;
	fn sqrt(self) -> Self::Output;
}

impl SquareRoot for f32 {
	type Output = Self;
	fn sqrt(self) -> Self {
		self.sqrt()
	}
}

impl SquareRoot for f64 {
	type Output = Self;
	fn sqrt(self) -> Self {
		self.sqrt()
	}
}
