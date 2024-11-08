"""
This module defines data structures used to represent sensor and positional information
in an autonomous vehicle context. It includes classes for handling various types of sensor
data such as velocity, orientation, motion, and positional data. It also includes
representations for image and point cloud data.

Classes:
    Velocity: Represents velocity data including timestamp, linear and angular velocities, and covariance.
    Heading: Represents heading/orientation data with timestamp and covariance.
    Motion: Represents motion data including orientation, angular velocity, and linear acceleration.
    Position: Represents position information including latitude, longitude, and altitude.
    Image: Represents an image with associated metadata, such as timestamp.
    Points: Represents a collection of 3D points with an associated timestamp.

Each class provides methods for initializing, serializing, and deserializing their respective
data types, as well as utility functions like converting timestamps to human-readable formats.
"""
from typing import Optional, Dict
from decimal import Decimal
from PIL import Image as PilImage
from aeifdataset.miscellaneous import read_data_block, TimestampMixin, ReprFormaterMixin
import numpy as np
from io import BytesIO


class Velocity(TimestampMixin, ReprFormaterMixin):
    """Class representing velocity data, including timestamp, linear and angular velocities, and covariance.

    Attributes:
        timestamp (Optional[Decimal]): The timestamp of the velocity measurement.
        linear_velocity (Optional[np.array]): The linear velocity vector.
        angular_velocity (Optional[np.array]): The angular velocity vector.
        covariance (Optional[np.array]): The covariance matrix of the velocity data.
    """

    def __init__(self, timestamp: Optional[Decimal] = None,
                 linear_velocity: Optional[np.array] = None,
                 angular_velocity: Optional[np.array] = None,
                 covariance: Optional[np.array] = None):
        """Initialize a Velocity object with timestamp, velocity vectors, and covariance.

        Args:
            timestamp (Optional[Decimal]): The timestamp of the velocity measurement.
            linear_velocity (Optional[np.array]): The linear velocity vector.
            angular_velocity (Optional[np.array]): The angular velocity vector.
            covariance (Optional[np.array]): The covariance matrix of the velocity data.
        """
        self.timestamp = timestamp
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity
        self.covariance = covariance

    def __repr__(self):
        """Return a string representation of the Velocity object with timestamp, linear, and angular velocities."""
        return (
            f"Velocity(\n"
            f"    timestamp={self.get_timestamp()},\n"
            f"    linear_velocity={self._format_array(self.linear_velocity)},\n"
            f"    angular_velocity={self._format_array(self.angular_velocity)}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class Heading(TimestampMixin, ReprFormaterMixin):
    """Class representing heading/orientation data with timestamp and covariance.

    Attributes:
        timestamp (Optional[Decimal]): The timestamp of the heading measurement.
        orientation (Optional[np.array]): The orientation vector.
        covariance (Optional[np.array]): The covariance matrix of the heading data.
    """

    def __init__(self, timestamp: Optional[Decimal] = None, orientation: Optional[np.array] = None,
                 covariance: Optional[np.array] = None):
        """Initialize a Heading object with timestamp, orientation, and covariance.

        Args:
            timestamp (Optional[Decimal]): The timestamp of the heading measurement.
            orientation (Optional[np.array]): The orientation vector.
            covariance (Optional[np.array]): The covariance matrix of the heading data.
        """
        self.timestamp = timestamp
        self.orientation = orientation
        self.covariance = covariance

    def __repr__(self):
        """Return a string representation of the Heading object with timestamp and orientation."""
        return (
            f"Heading(\n"
            f"    timestamp={self.get_timestamp()},\n"
            f"    orientation={self._format_array(self.orientation)}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class Motion(TimestampMixin, ReprFormaterMixin):
    """Class representing motion data including orientation, velocity, and acceleration.

    Attributes:
        timestamp (Optional[Decimal]): The timestamp of the motion measurement.
        orientation (Optional[np.array]): The orientation vector.
        orientation_covariance (Optional[np.array]): The covariance matrix of the orientation.
        angular_velocity (Optional[np.array]): The angular velocity vector.
        angular_velocity_covariance (Optional[np.array]): The covariance matrix of the angular velocity.
        linear_acceleration (Optional[np.array]): The linear acceleration vector.
        linear_acceleration_covariance (Optional[np.array]): The covariance matrix of the linear acceleration.
    """

    def __init__(self, timestamp: Optional[Decimal] = None, orientation: Optional[np.array] = None,
                 orientation_covariance: Optional[np.array] = None, angular_velocity: Optional[np.array] = None,
                 angular_velocity_covariance: Optional[np.array] = None, linear_acceleration: Optional[np.array] = None,
                 linear_acceleration_covariance: Optional[np.array] = None):
        """Initialize a Motion object with timestamp, orientation, velocity, and acceleration data.

        Args:
            timestamp (Optional[Decimal]): The timestamp of the motion measurement.
            orientation (Optional[np.array]): The orientation vector.
            orientation_covariance (Optional[np.array]): The covariance matrix of the orientation.
            angular_velocity (Optional[np.array]): The angular velocity vector.
            angular_velocity_covariance (Optional[np.array]): The covariance matrix of the angular velocity.
            linear_acceleration (Optional[np.array]): The linear acceleration vector.
            linear_acceleration_covariance (Optional[np.array]): The covariance matrix of the linear acceleration.
        """
        self.timestamp = timestamp
        self.orientation = orientation
        self.orientation_covariance = orientation_covariance
        self.angular_velocity = angular_velocity
        self.angular_velocity_covariance = angular_velocity_covariance
        self.linear_acceleration = linear_acceleration
        self.linear_acceleration_covariance = linear_acceleration_covariance

    def __repr__(self):
        """Return a string representation of the Motion object with key attributes."""

        return (
            f"Motion(\n"
            f"    timestamp={self.get_timestamp() if self.timestamp else 'None'},\n"
            f"    orientation={self._format_array(self.orientation)},\n"
            f"    angular_velocity={self._format_array(self.angular_velocity)},\n"
            f"    linear_acceleration={self._format_array(self.linear_acceleration)}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()


class Position(TimestampMixin, ReprFormaterMixin):
    """Class representing position information including latitude, longitude, and altitude.

    Attributes:
        timestamp (Optional[Decimal]): The timestamp of the position measurement.
        status (Optional[str]): The status of the GNSS signal.
        services (Optional[Dict[str, Optional[bool]]]): The status of satellite services (GPS, Glonass, etc.).
        latitude (Optional[Decimal]): The latitude in decimal degrees.
        longitude (Optional[Decimal]): The longitude in decimal degrees.
        altitude (Optional[Decimal]): The altitude in meters.
        covariance (Optional[np.array]): The covariance matrix of the position.
        covariance_type (Optional[str]): The type of covariance.
    """

    def __init__(self, timestamp: Optional[Decimal] = None, status: Optional[str] = None,
                 services: Optional[Dict[str, Optional[bool]]] = None, latitude: Optional[Decimal] = None,
                 longitude: Optional[Decimal] = None, altitude: Optional[Decimal] = None,
                 covariance: Optional[np.array] = None, covariance_type: Optional[str] = None):
        """Initialize a Position object with timestamp, location, and covariance data.

        Args:
            timestamp (Optional[Decimal]): The timestamp of the position measurement.
            status (Optional[str]): The status of the GNSS signal.
            services (Optional[Dict[str, Optional[bool]]]): The status of satellite services (GPS, Glonass, etc.).
            latitude (Optional[Decimal]): The latitude in decimal degrees.
            longitude (Optional[Decimal]): The longitude in decimal degrees.
            altitude (Optional[Decimal]): The altitude in meters.
            covariance (Optional[np.array]): The covariance matrix of the position.
            covariance_type (Optional[str]): The type of covariance.
        """
        self.timestamp = timestamp
        self.status = status
        self.services = self.init_services(services)
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.covariance = covariance
        self.covariance_type = covariance_type

    def __repr__(self):
        """Return a string representation of the Position object with latitude, longitude, and timestamp."""
        return (
            f"Position(\n"
            f"    timestamp={self.get_timestamp()},\n"
            f"    latitude={self.latitude:.8f},\n"
            f"    longitude={self.longitude:.8f}\n"
            f")"
        )

    def __str__(self):
        """Return the same representation as __repr__ for user-friendly output."""
        return self.__repr__()

    @staticmethod
    def init_services(services: Optional[Dict[str, Optional[bool]]]) -> Dict[str, Optional[bool]]:
        """Initialize the services dictionary with default values for GPS, Glonass, Galileo, and Baidou.

        Args:
            services (Optional[Dict[str, Optional[bool]]]): A dictionary with service statuses.

        Returns:
            Dict[str, Optional[bool]]: A dictionary with default values if not provided.
        """
        default_services = {'GPS': None, 'Glonass': None, 'Galileo': None, 'Baidou': None}
        if services is None:
            return default_services
        for key in default_services:
            services.setdefault(key, default_services[key])
        return services


class Image(TimestampMixin):
    """Class representing an image along with its metadata.

    Attributes:
        timestamp (Optional[Decimal]): Timestamp of the image.
        image (Optional[PilImage]): The actual image data.
    """

    def __init__(self, image: bytes = b"", timestamp: Optional[Decimal] = None):
        """Initialize an Image object with image data and a timestamp.

        Args:
            image (Optional[PilImage]): The image data.
            timestamp (Optional[Decimal]): Timestamp of the image.
        """
        self.image = image
        self.timestamp = timestamp

    def __getattr__(self, attr) -> PilImage:
        """
        Enables direct access to attributes of the `image` object.
        Parameters:
            attr (str): Name of the attribute to access.
        Returns:
            PilImage: Attribute value if it exists in the `image` object.
        Raises:
            AttributeError: If the attribute does not exist.
        """
        if hasattr(self.image, attr):
            return getattr(self.image, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def to_bytes(self) -> bytes:
        """Serialize the image to bytes using PNG compression.
        Returns:
            bytes: Serialized byte representation of the compressed image and timestamp.
        """
        img_byte_arr = BytesIO()
        self.image.save(img_byte_arr, format='JPEG', quality=85)
        encoded_img = img_byte_arr.getvalue()

        encoded_ts = str(self.timestamp).encode('utf-8')
        img_len = len(encoded_img).to_bytes(4, 'big')
        ts_len = len(encoded_ts).to_bytes(4, 'big')

        return img_len + encoded_img + ts_len + encoded_ts

    @classmethod
    def from_bytes(cls, data: bytes) -> 'Image':
        """Deserialize bytes to create an Image object from PNG or JPEG-compressed data.
        Args:
            data (bytes): The serialized byte data to deserialize.

        Returns:
            Image: The deserialized Image object.
        """
        img_bytes, data = read_data_block(data)
        ts_bytes, _ = read_data_block(data)

        img_instance = cls()
        img_instance.timestamp = Decimal(ts_bytes.decode('utf-8'))

        img_instance.image = img_bytes

        return img_instance


class Points(TimestampMixin):
    """Class representing a collection of points with an associated timestamp.

    Attributes:
        points (np.array): Array of points.
        timestamp (Decimal): Timestamp associated with the points.
    """

    def __init__(self, points: Optional[np.array] = None, timestamp: Optional[Decimal] = None):
        """Initialize a Points object with point data and a timestamp.

        Args:
            points (Optional[np.array]): Array of points.
            timestamp (Optional[Decimal]): Timestamp associated with the points.
        """
        self.points = points
        self.timestamp = timestamp

    def __getattr__(self, attr):
        """
        Enables direct access to attributes of the points object.

        Parameters:
            attr (str): Name of the attribute to access.

        Returns:
            np.array: Attribute value if it exists in the points object.

        Raises:
            AttributeError: If the attribute does not exist.
        """
        if hasattr(self.points, attr):
            return getattr(self.points, attr)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __getitem__(self, index: int) -> np.array:
        """Enable subscriptable access to the points array.

        Args:
            index (int): Index to access the points array.

        Returns:
            np.array: The indexed points data.
        """
        if self.points is not None and self.points is not None:
            return self.points[index]
        raise IndexError(f"'{type(self).__name__}' object has no points data to index.")

    def to_bytes(self) -> bytes:
        """Serialize the points data to bytes.

        Returns:
            bytes: Serialized byte representation of the points and timestamp.
        """
        encoded_pts = self.points.tobytes()
        encoded_ts = str(self.timestamp).encode('utf-8')
        pts_len = len(encoded_pts).to_bytes(4, 'big')
        ts_len = len(encoded_ts).to_bytes(4, 'big')
        return pts_len + encoded_pts + ts_len + encoded_ts

    @classmethod
    def from_bytes(cls, data: bytes, dtype: np.dtype) -> 'Points':
        """Deserialize bytes to create a Points object.

        Args:
            data (bytes): The serialized byte data to deserialize.
            dtype (np.dtype): The data type of the points.

        Returns:
            Points: The deserialized Points object.
        """
        pts_bytes, data = read_data_block(data)
        ts_bytes, _ = read_data_block(data)
        pts_instance = cls()
        pts_instance.timestamp = Decimal(ts_bytes.decode('utf-8'))
        pts_instance.points = np.frombuffer(pts_bytes, dtype=dtype)
        return pts_instance
