//! Representations of the real and complex numbers.

use std::ops::{Deref, DerefMut, Add, Sub, Mul, Div, Neg, Rem};
use std::cmp::{Ordering, PartialOrd, Ord};

pub trait AdditiveIdentity {
    fn zero() -> Self;
}

pub trait MultiplicativeIdentity {
    fn one() -> Self;
}

pub trait Trigonometric: Copy + PartialEq + AdditiveIdentity + MultiplicativeIdentity + Div<Output=Self> {

    fn sin(self) -> Self;

    fn cos(self) -> Self;

    fn tan(self) -> Option<Self> {
        if self.cos() == Self::zero() {
            return None;
        }
        Some(self.sin() / self.cos())
    }

    fn csc(self) -> Option<Self> {
        if self.sin() == Self::zero() {
            return None;
        }
        Some(Self::one() / self.sin())
    }

    fn sec(self) -> Option<Self> {
        if self.cos() == Self::zero() {
            return None;
        }
        Some(Self::one() / self.cos())
    }

    fn cot(self) -> Option<Self> {
        if self.sin() == Self::zero() {
            return None;
        }
        Some(self.cos() / self.sin())
    }

}

pub trait InverseTrigonometric: Copy + PartialEq + AdditiveIdentity + Div<Output=Self> {

    fn arcsin(self) -> Self;

    fn arccos(self) -> Self;

    fn arctan(self) -> Self;

    fn arccsc(self) -> Option<Self>;

    fn arcsec(self) -> Option<Self>;

    fn arccot(self) -> Option<Self>;

}

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

    const EQUALITY_THRESHOLD: f64 = 3.0*std::f64::EPSILON;
    const APPROXIMATION_ACCURACY: usize = 5;    

    #[inline]
    pub const fn new(value: f64) -> Self {
        Self { value }
    }

    #[inline]
    pub fn is_positive(&self) -> bool {
        self > &real![0]
    }

    #[inline]
    pub fn is_negative(&self) -> bool {
        self < &real![0]
    }

    fn approx_eq(a: f64, b: f64) -> bool {
        (a - b).abs() <= Self::EQUALITY_THRESHOLD
    }

    #[inline]
    pub fn is_integer(&self) -> bool {
        real![self.value.fract()] == real![0]
    }

    #[inline]
    pub fn is_fractional(&self) -> bool {
        !self.is_integer()
    }

    fn sign(n: f64) -> isize {
        if n >= 0.0 {
            return 1;
        }
        return -1;
    }

    fn gcd(mut a: usize, mut b: usize) -> usize {
        while b > 0 {
            let overflow = a % b;
            a = b;
            b = overflow;
        }
        a
    }

    pub fn to_rational(&self) -> (isize, usize) {
        let base = self.value.log2().floor();
        if base >= 0.0 {
            return (self.value as isize, 1)
        }
        let numerator = (self.value / f64::EPSILON) as usize;
        let denominator = f64::EPSILON.powi(-1) as usize;
        let gcd = Self::gcd(numerator, denominator);
        ((numerator / gcd) as isize * Self::sign(self.value), denominator / gcd)
    }

    pub fn fact(&self) -> Self {
        if self.is_fractional() {
            todo!("Support for the Gamma function has not yet been implemented")
        }
        if self == &real![0] {
            return real![1];
        }
        real![(1..=(self.value as usize)).product::<usize>()]
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
        Self::approx_eq(self.value, other.value)
    }
}

impl Eq for Real {}

impl PartialOrd<Self> for Real {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.value < other.value {
            return Some(Ordering::Less);
        }
        if self.value > other.value {
            return Some(Ordering::Greater);
        }
        Some(Ordering::Equal)
    }
}

impl Ord for Real {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

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

impl AdditiveIdentity for Real {
    fn zero() -> Self {
        real![0]
    }
}

impl MultiplicativeIdentity for Real {
    fn one() -> Self {
        real![1]
    }
}

impl Trigonometric for Real {

    fn sin(self) -> Self {
        real![self.value.sin()]
    }

    fn cos(self) -> Self {
        real![self.value.cos()]
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
        if Self::approx_eq(other.value, 0.0) {
            panic!("Cannot attempt division by zero")
        }
        Self::new(self.value / other.value)
    }
}

impl Neg for Real {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.value)
    }
}

impl Rem<Self> for Real {
    type Output = Self;
    fn rem(self, other: Self) -> Self {
        if self.is_fractional() && other.is_fractional() {
            todo!("Modular arithmetic over the reals is as-of-yet unimplemented")
        }
        if self.is_fractional() {
            todo!("Modular arithmetic over the reals is as-of-yet unimplemented")
        }
        if other.is_fractional() {
            todo!("Modular arithmetic over the reals is as-of-yet unimplemented")
        }
        if other == real![0] {
            return self;
        }
        Self::new(self.value % other.value)
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
        fn left_absorption() {
            assert_eq!(real![0] * real![134], real![0])
        }

        #[test]
        fn right_absorption() {
            assert_eq!(real![1431] * real![0], real![0])
        }

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

    mod division {

        use super::*;

        #[test]
        fn left_absorption() {
            assert_eq!(real![0] / real![12], real![0])
        }

        #[test]
        #[should_panic]
        fn divide_by_zero() {
            assert_eq!(real![12] / real![0], real![std::f64::NAN])
        }

        #[test]
        fn left_invertibility() {
            assert_eq!((real![6] / real![12]) * real![2], real![1])
        }

        #[test]
        fn right_identity() {
            assert_eq!(real![12] / real![1], real![12])
        }

        #[test]
        fn double_positive() {
            assert_eq!(real![12] / real![3], real![4])
        }

        #[test]
        fn mixed_sign() {
            assert_eq!(real![12] / real![-2], real![-6]);
            assert_eq!(real![-12] / real![2], real![-6])
        }

        #[test]
        fn double_negative() {
            assert_eq!(real![-12] / real![-3], real![4])
        }

        #[test]
        fn fractional() {
            assert_eq!(real![12.6] / real![-6.3], real![-2])
        }

    }

    mod negation {

        use super::*;

        #[test]
        fn zero() {
            assert_eq!(real![0], -real![0])
        }

        #[test]
        fn positive() {
            assert_eq!(real![-1], -real![1])
        }

        #[test]
        fn negative() {
            assert_eq!(real![1], -real![-1])
        }

        #[test]
        fn fractional() {
            assert_eq!(real![121.2], -real![-121.2])
        }

    }

    mod remainder {

        use super::*;

        #[test]
        fn mod_zero() {
            assert_eq!(real![12] % real![0], real![12])
        }

        #[test]
        fn mod_one() {
            assert_eq!(real![12] % real![1], real![0])
        }

        #[test]
        fn max() {
            assert_eq!(real![13] % real![13], real![0])
        }

        #[test]
        fn min() {
            assert_eq!(real![0] % real![1321], real![0])
        }

        #[test]
        fn standard() {
            assert_eq!(real![13] % real![6], real![1])
        }

        #[ignore]
        #[test]
        fn float_mod_int() {
            assert_eq!(real![2.5] % real![2], real![0.5])
        }

        #[ignore]
        #[test]
        fn float_mod_float() {
            assert_eq!(real![3.5] % real![0.5], real![7]);
            assert_eq!(real![3.5] % real![2.5], real![1])
        }

        #[ignore]
        #[test]
        fn int_mod_float() {
            assert_eq!(real![2] % real![0.5], real![4])
        }

    }

    mod factorials {

        use super::*;

        #[test]
        fn zero() {
            assert_eq!(real![0].fact(), real![1])
        }

        #[test]
        fn one() {
            assert_eq!(real![1].fact(), real![1])
        }

        #[test]
        fn standard() {
            assert_eq!(real![5].fact(), real![120])
        }

        #[ignore]
        #[should_panic]
        #[test]
        fn negative_int() {
            let _ = real![-3].fact();
        }

        #[ignore]
        #[test]
        fn fractional() {
            assert_eq!(real![0.5].fact(), real![0.5*std::f64::consts::PI.sqrt()]);
            assert_eq!(real![-1.5].fact(), real![-2.0*std::f64::consts::PI.sqrt()])
        }

    }

}