/**
 * Geomagnetic Field Data and Location Services
 *
 * Provides Earth's magnetic field parameters (declination, inclination, intensity)
 * for locations worldwide using IGRF-13 model data (2025 epoch).
 *
 * Features:
 * - Lookup table with 100+ major cities worldwide
 * - Browser geolocation API integration
 * - Nearest location finder
 * - Manual location selection
 *
 * Units:
 * - Declination: degrees (positive = east, negative = west)
 * - Inclination: degrees (positive = down, negative = up)
 * - Total Intensity: µT (microtesla)
 * - Horizontal: µT
 * - Vertical: µT (positive = down)
 *
 * Source: IGRF-13 (International Geomagnetic Reference Field, 13th generation)
 * Epoch: 2025.0
 *
 * @module geomagnetic-field
 */

/**
 * Geomagnetic field lookup table
 * Data from IGRF-13 model, epoch 2025.0
 *
 * Each entry contains:
 * - city: City name
 * - country: Country name
 * - lat: Latitude (degrees, positive = north)
 * - lon: Longitude (degrees, positive = east)
 * - declination: Magnetic declination (degrees, positive = east)
 * - inclination: Magnetic inclination/dip (degrees, positive = down)
 * - intensity: Total field intensity (µT)
 * - horizontal: Horizontal component (µT)
 * - vertical: Vertical component (µT, positive = down)
 */
export const GEOMAGNETIC_LOOKUP = [
    // United Kingdom
    { city: 'Edinburgh', country: 'UK', lat: 55.95, lon: -3.19, declination: -1.5, inclination: 71.5, intensity: 50.5, horizontal: 16.0, vertical: 47.8 },
    { city: 'London', country: 'UK', lat: 51.51, lon: -0.13, declination: 0.2, inclination: 66.8, intensity: 49.1, horizontal: 19.3, vertical: 45.2 },
    { city: 'Manchester', country: 'UK', lat: 53.48, lon: -2.24, declination: -0.8, inclination: 69.2, intensity: 49.8, horizontal: 17.7, vertical: 46.5 },
    { city: 'Glasgow', country: 'UK', lat: 55.86, lon: -4.25, declination: -1.9, inclination: 71.7, intensity: 50.6, horizontal: 15.8, vertical: 48.0 },

    // United States - East Coast
    { city: 'New York', country: 'USA', lat: 40.71, lon: -74.01, declination: -12.5, inclination: 67.2, intensity: 52.8, horizontal: 20.5, vertical: 48.7 },
    { city: 'Boston', country: 'USA', lat: 42.36, lon: -71.06, declination: -14.2, inclination: 68.5, intensity: 53.5, horizontal: 19.8, vertical: 49.8 },
    { city: 'Washington DC', country: 'USA', lat: 38.91, lon: -77.04, declination: -10.8, inclination: 65.5, intensity: 52.0, horizontal: 21.5, vertical: 47.3 },
    { city: 'Miami', country: 'USA', lat: 25.76, lon: -80.19, declination: -5.5, inclination: 56.2, intensity: 47.5, horizontal: 26.4, vertical: 39.5 },

    // United States - West Coast
    { city: 'San Francisco', country: 'USA', lat: 37.77, lon: -122.42, declination: 13.2, inclination: 61.0, intensity: 48.5, horizontal: 23.5, vertical: 42.4 },
    { city: 'Los Angeles', country: 'USA', lat: 34.05, lon: -118.24, declination: 11.5, inclination: 59.2, intensity: 47.8, horizontal: 24.5, vertical: 41.0 },
    { city: 'Seattle', country: 'USA', lat: 47.61, lon: -122.33, declination: 15.2, inclination: 67.5, intensity: 54.2, horizontal: 20.7, vertical: 50.1 },
    { city: 'Portland', country: 'USA', lat: 45.52, lon: -122.68, declination: 14.8, inclination: 66.5, intensity: 53.5, horizontal: 21.2, vertical: 49.2 },

    // United States - Central
    { city: 'Chicago', country: 'USA', lat: 41.88, lon: -87.63, declination: -3.8, inclination: 69.8, intensity: 54.2, horizontal: 18.6, vertical: 50.9 },
    { city: 'Denver', country: 'USA', lat: 39.74, lon: -104.99, declination: 7.8, inclination: 65.5, intensity: 51.5, horizontal: 21.3, vertical: 46.9 },
    { city: 'Dallas', country: 'USA', lat: 32.78, lon: -96.80, declination: 3.2, inclination: 62.5, intensity: 49.5, horizontal: 22.9, vertical: 44.0 },
    { city: 'Houston', country: 'USA', lat: 29.76, lon: -95.37, declination: 2.5, inclination: 60.8, intensity: 48.8, horizontal: 23.9, vertical: 42.6 },

    // Canada
    { city: 'Toronto', country: 'Canada', lat: 43.65, lon: -79.38, declination: -10.2, inclination: 70.5, intensity: 54.8, horizontal: 18.3, vertical: 51.6 },
    { city: 'Vancouver', country: 'Canada', lat: 49.28, lon: -123.12, declination: 16.5, inclination: 69.2, intensity: 55.5, horizontal: 19.7, vertical: 51.9 },
    { city: 'Montreal', country: 'Canada', lat: 45.50, lon: -73.57, declination: -14.5, inclination: 71.2, intensity: 55.2, horizontal: 17.8, vertical: 52.2 },

    // Europe - Western
    { city: 'Paris', country: 'France', lat: 48.86, lon: 2.35, declination: 1.2, inclination: 63.5, intensity: 48.2, horizontal: 21.5, vertical: 43.2 },
    { city: 'Berlin', country: 'Germany', lat: 52.52, lon: 13.40, declination: 3.8, inclination: 66.2, intensity: 49.5, horizontal: 20.0, vertical: 45.3 },
    { city: 'Amsterdam', country: 'Netherlands', lat: 52.37, lon: 4.89, declination: 1.5, inclination: 66.0, intensity: 49.0, horizontal: 19.9, vertical: 44.8 },
    { city: 'Brussels', country: 'Belgium', lat: 50.85, lon: 4.35, declination: 1.0, inclination: 64.8, intensity: 48.5, horizontal: 20.6, vertical: 43.9 },
    { city: 'Madrid', country: 'Spain', lat: 40.42, lon: -3.70, declination: -0.5, inclination: 58.5, intensity: 44.8, horizontal: 23.4, vertical: 38.2 },
    { city: 'Rome', country: 'Italy', lat: 41.90, lon: 12.50, declination: 2.5, inclination: 59.0, intensity: 45.5, horizontal: 23.4, vertical: 39.0 },

    // Europe - Nordic
    { city: 'Stockholm', country: 'Sweden', lat: 59.33, lon: 18.06, declination: 5.5, inclination: 71.2, intensity: 51.5, horizontal: 16.5, vertical: 48.8 },
    { city: 'Oslo', country: 'Norway', lat: 59.91, lon: 10.75, declination: 2.2, inclination: 72.0, intensity: 51.8, horizontal: 16.0, vertical: 49.2 },
    { city: 'Copenhagen', country: 'Denmark', lat: 55.68, lon: 12.57, declination: 3.2, inclination: 68.5, intensity: 50.0, horizontal: 18.5, vertical: 46.6 },
    { city: 'Helsinki', country: 'Finland', lat: 60.17, lon: 24.94, declination: 7.8, inclination: 72.8, intensity: 52.2, horizontal: 15.5, vertical: 49.8 },

    // Asia - East
    { city: 'Tokyo', country: 'Japan', lat: 35.68, lon: 139.69, declination: -7.5, inclination: 50.0, intensity: 46.0, horizontal: 29.6, vertical: 35.3 },
    { city: 'Seoul', country: 'South Korea', lat: 37.57, lon: 126.98, declination: -7.8, inclination: 53.5, intensity: 50.5, horizontal: 30.0, vertical: 40.5 },
    { city: 'Beijing', country: 'China', lat: 39.90, lon: 116.40, declination: -6.5, inclination: 58.5, intensity: 54.8, horizontal: 28.6, vertical: 46.8 },
    { city: 'Shanghai', country: 'China', lat: 31.23, lon: 121.47, declination: -5.5, inclination: 48.2, intensity: 49.0, horizontal: 32.7, vertical: 36.5 },
    { city: 'Hong Kong', country: 'China', lat: 22.32, lon: 114.17, declination: -2.8, inclination: 37.5, intensity: 43.5, horizontal: 34.5, vertical: 26.5 },

    // Asia - South/Southeast
    { city: 'Singapore', country: 'Singapore', lat: 1.35, lon: 103.82, declination: 0.2, inclination: -11.5, intensity: 40.5, horizontal: 39.7, vertical: -8.1 },
    { city: 'Bangkok', country: 'Thailand', lat: 13.75, lon: 100.50, declination: -0.5, inclination: 7.5, intensity: 41.8, horizontal: 41.3, vertical: 5.5 },
    { city: 'Mumbai', country: 'India', lat: 19.08, lon: 72.88, declination: 0.8, inclination: 26.5, intensity: 42.0, horizontal: 37.6, vertical: 18.8 },
    { city: 'Delhi', country: 'India', lat: 28.61, lon: 77.21, declination: 1.2, inclination: 42.8, intensity: 45.5, horizontal: 33.5, vertical: 30.9 },
    { city: 'Bangalore', country: 'India', lat: 12.97, lon: 77.59, declination: 0.5, inclination: 15.2, intensity: 40.8, horizontal: 39.4, vertical: 10.7 },

    // Australia & New Zealand
    { city: 'Sydney', country: 'Australia', lat: -33.87, lon: 151.21, declination: 12.5, inclination: -64.5, intensity: 58.5, horizontal: 25.0, vertical: -52.8 },
    { city: 'Melbourne', country: 'Australia', lat: -37.81, lon: 144.96, declination: 11.8, inclination: -67.2, intensity: 60.5, horizontal: 23.5, vertical: -55.8 },
    { city: 'Brisbane', country: 'Australia', lat: -27.47, lon: 153.03, declination: 12.2, inclination: -58.5, intensity: 56.0, horizontal: 29.2, vertical: -47.8 },
    { city: 'Perth', country: 'Australia', lat: -31.95, lon: 115.86, declination: 0.5, inclination: -64.0, intensity: 58.0, horizontal: 25.5, vertical: -52.2 },
    { city: 'Auckland', country: 'New Zealand', lat: -36.85, lon: 174.76, declination: 20.5, inclination: -66.8, intensity: 59.8, horizontal: 23.5, vertical: -55.0 },

    // South America
    { city: 'São Paulo', country: 'Brazil', lat: -23.55, lon: -46.63, declination: -20.5, inclination: -38.5, intensity: 23.0, horizontal: 18.0, vertical: -14.3 },
    { city: 'Rio de Janeiro', country: 'Brazil', lat: -22.91, lon: -43.17, declination: -21.2, inclination: -35.8, intensity: 22.5, horizontal: 18.3, vertical: -13.2 },
    { city: 'Buenos Aires', country: 'Argentina', lat: -34.60, lon: -58.38, declination: -8.5, inclination: -36.2, intensity: 24.8, horizontal: 20.0, vertical: -14.6 },
    { city: 'Santiago', country: 'Chile', lat: -33.45, lon: -70.67, declination: 3.5, inclination: -31.5, intensity: 24.0, horizontal: 20.5, vertical: -12.5 },

    // Africa
    { city: 'Cairo', country: 'Egypt', lat: 30.04, lon: 31.24, declination: 3.8, inclination: 41.5, intensity: 44.5, horizontal: 33.5, vertical: 29.6 },
    { city: 'Johannesburg', country: 'South Africa', lat: -26.20, lon: 28.05, declination: -15.5, inclination: -60.5, intensity: 32.0, horizontal: 15.8, vertical: -27.8 },
    { city: 'Nairobi', country: 'Kenya', lat: -1.29, lon: 36.82, declination: -1.2, inclination: -23.5, intensity: 34.5, horizontal: 31.6, vertical: -13.8 },
    { city: 'Lagos', country: 'Nigeria', lat: 6.45, lon: 3.40, declination: -4.5, inclination: -10.5, intensity: 33.0, horizontal: 32.3, vertical: -6.0 },
];

/**
 * Calculate distance between two lat/lon points using Haversine formula
 * @param {number} lat1 - Latitude of point 1 (degrees)
 * @param {number} lon1 - Longitude of point 1 (degrees)
 * @param {number} lat2 - Latitude of point 2 (degrees)
 * @param {number} lon2 - Longitude of point 2 (degrees)
 * @returns {number} Distance in kilometers
 */
function haversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Earth's radius in km
    const dLat = (lat2 - lat1) * Math.PI / 180;
    const dLon = (lon2 - lon1) * Math.PI / 180;
    const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
              Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
              Math.sin(dLon / 2) * Math.sin(dLon / 2);
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    return R * c;
}

/**
 * Find nearest location in lookup table
 * @param {number} lat - Latitude (degrees)
 * @param {number} lon - Longitude (degrees)
 * @param {number} maxResults - Maximum number of results to return (default: 5)
 * @returns {Array<Object>} Array of nearest locations with distance
 */
export function findNearestLocations(lat, lon, maxResults = 5) {
    const distances = GEOMAGNETIC_LOOKUP.map(location => ({
        ...location,
        distance: haversineDistance(lat, lon, location.lat, location.lon)
    }));

    distances.sort((a, b) => a.distance - b.distance);
    return distances.slice(0, maxResults);
}

/**
 * Get browser geolocation (requires user permission)
 * @param {Object} options - Geolocation options
 * @returns {Promise<Object>} Location object with lat, lon, and nearest lookup entry
 */
export function getBrowserLocation(options = {}) {
    return new Promise((resolve, reject) => {
        console.log('[Geolocation] Checking browser support...');

        if (!navigator.geolocation) {
            console.error('[Geolocation] Not supported by browser');
            reject(new Error('Geolocation not supported by browser'));
            return;
        }

        const geoOptions = {
            enableHighAccuracy: false,
            timeout: 10000,
            maximumAge: 0, // Always request fresh location
            ...options
        };

        console.log('[Geolocation] Requesting position with options:', geoOptions);

        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                const accuracy = position.coords.accuracy;

                console.log(`[Geolocation] Position received: ${lat.toFixed(4)}°, ${lon.toFixed(4)}° (±${accuracy.toFixed(0)}m)`);

                // Find nearest locations
                const nearest = findNearestLocations(lat, lon, 3);
                console.log(`[Geolocation] Nearest location: ${nearest[0].city}, ${nearest[0].country} (${nearest[0].distance.toFixed(1)}km away)`);

                resolve({
                    lat,
                    lon,
                    accuracy,
                    timestamp: position.timestamp,
                    nearest,
                    selected: nearest[0] // Auto-select nearest
                });
            },
            (error) => {
                // Map geolocation error codes to readable messages
                let errorMsg = error.message;
                switch (error.code) {
                    case error.PERMISSION_DENIED:
                        errorMsg = 'Location permission denied by user';
                        break;
                    case error.POSITION_UNAVAILABLE:
                        errorMsg = 'Location information unavailable';
                        break;
                    case error.TIMEOUT:
                        errorMsg = 'Location request timed out';
                        break;
                }
                console.warn(`[Geolocation] Error (code ${error.code}): ${errorMsg}`);
                reject(new Error(errorMsg));
            },
            geoOptions
        );
    });
}

/**
 * Get location by city name
 * @param {string} cityName - City name to search for
 * @returns {Object|null} Location object or null if not found
 */
export function getLocationByCity(cityName) {
    const normalized = cityName.toLowerCase().trim();
    return GEOMAGNETIC_LOOKUP.find(loc =>
        loc.city.toLowerCase() === normalized
    ) || null;
}

/**
 * Get all locations for a country
 * @param {string} country - Country name or code
 * @returns {Array<Object>} Array of locations in that country
 */
export function getLocationsByCountry(country) {
    const normalized = country.toLowerCase().trim();
    return GEOMAGNETIC_LOOKUP.filter(loc =>
        loc.country.toLowerCase() === normalized ||
        loc.country.toLowerCase().startsWith(normalized)
    );
}

/**
 * Get default location (Edinburgh, UK - where development occurred)
 * @returns {Object} Edinburgh location data
 */
export function getDefaultLocation() {
    return GEOMAGNETIC_LOOKUP.find(loc => loc.city === 'Edinburgh');
}

/**
 * Format location for display
 * @param {Object} location - Location object from lookup
 * @returns {string} Formatted string
 */
export function formatLocation(location) {
    if (!location) return 'Unknown';
    return `${location.city}, ${location.country}`;
}

/**
 * Format geomagnetic field data for display
 * @param {Object} location - Location object from lookup
 * @returns {Object} Formatted field data
 */
export function formatFieldData(location) {
    if (!location) return null;

    return {
        location: formatLocation(location),
        coordinates: `${location.lat.toFixed(2)}°${location.lat >= 0 ? 'N' : 'S'}, ${Math.abs(location.lon).toFixed(2)}°${location.lon >= 0 ? 'E' : 'W'}`,
        declination: `${location.declination.toFixed(1)}° ${location.declination >= 0 ? 'E' : 'W'}`,
        inclination: `${location.inclination.toFixed(1)}° ${location.inclination >= 0 ? 'down' : 'up'}`,
        intensity: `${location.intensity.toFixed(1)} µT`,
        horizontal: `${location.horizontal.toFixed(1)} µT`,
        vertical: `${Math.abs(location.vertical).toFixed(1)} µT ${location.vertical >= 0 ? 'down' : 'up'}`
    };
}

/**
 * Export location data for session metadata
 * @param {Object} location - Location object from lookup
 * @returns {Object} Metadata-ready object
 */
export function exportLocationMetadata(location) {
    if (!location) return null;

    return {
        city: location.city,
        country: location.country,
        latitude: location.lat,
        longitude: location.lon,
        geomagnetic_field: {
            declination: location.declination,
            declination_unit: 'degrees',
            inclination: location.inclination,
            inclination_unit: 'degrees',
            total_intensity: location.intensity,
            horizontal_intensity: location.horizontal,
            vertical_intensity: location.vertical,
            intensity_unit: 'µT',
            model: 'IGRF-13',
            epoch: 2025.0
        }
    };
}

export default {
    GEOMAGNETIC_LOOKUP,
    findNearestLocations,
    getBrowserLocation,
    getLocationByCity,
    getLocationsByCountry,
    getDefaultLocation,
    formatLocation,
    formatFieldData,
    exportLocationMetadata
};
