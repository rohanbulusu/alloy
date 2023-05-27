//! Representations of the real and complex numbers.

use std::ops::{Deref, DerefMut, Add, Sub, Mul, Div, Neg};

macro_rules! real {
    ($val:expr) => {{
        let value: Real = $val.into();
        value
    }}
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Real {
    value: f64
}

impl Real {

    pub const EQUALITY_THRESHOLD: f64 = 3.0*std::f64::EPSILON;
    
    #[inline]
    pub const fn new(value: f64) -> Self {
        Self { value }
    }
}

macro_rules! real_from {
    ($t:ty) => {
        impl From<$t> for Real {
            fn from(value: $t) -> Self {
                Self::new(value as f64)
            }
        }
    }
}

real_from!(u8);
real_from!(u16);
real_from!(u32);
real_from!(u64);

real_from!(i8);
real_from!(i16);
real_from!(i32);
real_from!(i64);

real_from!(f32);
real_from!(f64);

real_from!(usize);
real_from!(isize);


unsafe impl Send for Real {}

unsafe impl Sync for Real {}

impl PartialEq for Real {
    fn eq(&self, other: &Self) -> bool {
        (self.value - other.value).abs() <= Self::EQUALITY_THRESHOLD
    }
}

impl Eq for Real {}

impl std::fmt::Display for Real {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Real({})", self.value)
    }
}

impl std::fmt::Debug for Real {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Real({})", self.value)
    }
}

impl Deref for Real {
    type Target = f64;
    fn deref(&self) -> &f64 {
        &self.value
    }
}

impl DerefMut for Real {
    fn deref_mut(&mut self) -> &mut f64 {
        &mut self.value
    }
}

impl Add<Self> for Real {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self::new(self.value + other.value)
    }
}

impl Sub<Self> for Real {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self::new(self.value - other.value)
    }
}

impl Mul<Self> for Real {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self::new(self.value * other.value)
    }
}

impl Div<Self> for Real {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        Self::new(self.value / other.value)
    }
}

impl Neg for Real {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.value)
    }
}

#[cfg(test)]
mod real {

    use super::*;

    mod eq {

        use super::*;

        #[test]
        fn equality() {
            assert_eq!(real![3], real![3])
        }

        #[test]
        fn inequality() {
            assert_ne!(real![12], real![-3])
        }

        #[test]
        fn opposites() {
            assert_ne!(real![5], real![-5])
        }

    }

    mod addition {

        use super::*;

        #[test]
        fn left_identity() {
            assert_eq!(real![0] + real![12], real![12])
        }

        #[test]
        fn right_identity() {
            assert_eq!(real![12] + real![0], real![12])
        }

        #[test]
        fn double_positive() {
            assert_eq!(real![13] + real![23], real![36])
        }

        #[test]
        fn mixed_sign() {
            assert_eq!(real![23] + real![-12], real![11])
        }

        #[test]
        fn double_negative() {
            assert_eq!(real![-12] + real![-32], real![-44])
        }

        #[test]
        fn fractional() {
            assert_eq!(real![-12.3] + real![13.5], real![1.2])
        }

    }

    mod subtraction {

        use super::*;

        #[test]
        fn left_identity() {
            assert_eq!(real![0] - real![12], real![-12])
        }

        #[test]
        fn right_identity() {
            assert_eq!(real![12] - real![0], real![12])
        }

        #[test]
        fn double_positive() {
            assert_eq!(real![13] - real![23], real![-10])
        }

        #[test]
        fn mixed_sign() {
            assert_eq!(real![23] - real![-12], real![35])
        }

        #[test]
        fn double_negative() {
            assert_eq!(real![-12] - real![-32], real![20])
        }

        #[test]
        fn fractional() {
            assert_eq!(real![-12.3] - real![13.5], real![-25.8])
        }

    }

    mod multiplication {

        use super::*;

        #[test]
        fn left_identity() {
            assert_eq!(real![1] * real![12], real![12])
        }

        #[test]
        fn right_identity() {
            assert_eq!(real![12] * real![1], real![12])
        }

        #[test]
        fn double_positive() {
            assert_eq!(real![2] * real![3], real![6])
        }

        #[test]
        fn mixed_sign() {
            assert_eq!(real![23] * real![-4], real![-92])
        }

        #[test]
        fn double_negative() {
            assert_eq!(real![-12] * real![-12], real![144])
        }

        #[test]
        fn fractional() {
            assert_eq!(real![-1.5] * real![12.6], real![-18.9])
        }

    }

}