//! Structs for subatomic particles and their mechanics.

use space::{Vector, Point3};

/// Speed of light in m/s
pub const C: usize = 299792458;

/// Planck's constant in J/Hz
pub const H: f64 = 6.62607015e-34;


struct PhotonEnergy {
	pub frequency: f32,
	pub joules: f32
}

impl PhotonEnergy {

	fn frequency_to_joules(frequency: f32) -> f32 {
		(C as f32)*(H as f32)*frequency
	}

	fn wavelength_to_joules(wavelength: f32) -> f32 {
		(C as f32)*(H as f32) / wavelength
	}

	fn wavelength_to_frequency(wavelength: f32) -> f32 {
		1.0 / wavelength
	}

	pub fn with_frequency(frequency: f32) -> Self {
		Self { 
			frequency,
			joules: Self::frequency_to_joules(frequency)
		}
	}

	pub fn with_wavelength(wavelength: f32) -> Self {
		Self {
			frequency: Self::wavelength_to_frequency(wavelength),
			joules: Self::wavelength_to_joules(wavelength)
		}
	}

}


/// Model for a photon.
pub struct Photon {
	position: Point3,
	velocity: Vector<f32>,
	energy: PhotonEnergy
}

impl Photon {

	/// Default frequency for a `Photon` constructed with [`Photon::new`].
	pub const DEFAULT_PHOTON_FREQUENCY: f32 = 594.5;

	/// Constructs a new `Photon` from a [`Point3`] position and a [`Vector`] 
	/// velocity.
	pub fn new(position: Point3, velocity: Vector<f32>) -> Self {
		Self {
			position,
			velocity: velocity*(C as f32),
			energy: PhotonEnergy::with_frequency(Self::DEFAULT_PHOTON_FREQUENCY)
		}
	}

	/// Constructs a new `Photon` from a [`Point3`] position, a [`Vector`]
	/// velocity, and an [`f32`] frequency.
	pub fn with_frequency(position: Point3, velocity: Vector<f32>, frequency: f32) -> Self {
		Self {
			position,
			velocity: velocity*(C as f32),
			energy: PhotonEnergy::with_frequency(frequency)
		}
	}

	/// Constructs a new `Photon` from a [`Point3`] position, a [`Vector`]
	/// velocity, and an [`f32`] wavelength.
	pub fn with_wavelength(position: Point3, velocity: Vector<f32>, wavelength: f32) -> Self {
		Self {
			position,
			velocity: velocity*(C as f32),
			energy: PhotonEnergy::with_wavelength(wavelength)
		}
	}

	/// Computes the density of the probability cloud of `self` without regard
	/// to the presence of potentially interacting surrounding particles.
	pub fn raw_probability_at(&self, location: Point3) -> f32 {
		let to_center = <Point3 as Into<Vector<f32>>>::into(location) - self.position.into();
		to_center.norm().exp().recip()
	}

}

