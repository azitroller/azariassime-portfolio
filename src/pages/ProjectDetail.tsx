import { useParams, useNavigate } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { ArrowLeft, Calendar, User, ChevronRight } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import Navigation from '@/components/Navigation';
import Footer from '@/components/Footer';
import ContactSection from '@/components/ContactSection';
import { CodePreview } from '@/components/CodePreview';
import { MathEquation } from '@/components/MathEquation';

// Enhanced project data structure for authoritative content
const projectsData = {
  "automated-valve-test": {
    id: "automated-valve-test",
    title: "Automated Valve Test Platform for High-Pressure Systems",
    subtitle: "Python-driven automation system for aerospace grade valve testing",
    category: "Test Automation",
    date: "2024",
    author: "Aerospace Engineering Team", 
    tags: ["High-Pressure Testing", "Python", "Automation"],
    hero: "/lovable-uploads/000f98ca-15f2-4d60-a820-a33b989ababe.png",
    sections: [
      {
        type: "text-left",
        title: "Project Overview",
        content: `This project focused on developing a comprehensive automated testing platform for high-pressure aerospace valve systems using Python-driven automation. The system was designed to test valves under extreme conditions including pressures up to 6000 psi and temperatures ranging from -40°C to 150°C.

        The primary objective was to create a reliable, repeatable testing framework that could automatically cycle through various test parameters while continuously monitoring valve performance metrics. This automation significantly reduced testing time from manual operations while improving data accuracy and repeatability.`,
        visual: {
          type: "terminal",
          content: `// Test Parameters
Max Pressure: 6000 psi
Temperature Range: -40°C to 150°C
Cycle Count: 10,000 cycles
Data Sampling Rate: 1000 Hz
Test Duration: 72 hours

// Valve Specifications
Valve Type: Ball valve, Gate valve
Port Size: 1/4" to 2"
Material: 316L Stainless Steel
Seal Type: PTFE, Viton`
        }
      },
      {
        type: "text-right", 
        title: "System Architecture",
        content: `The automated test platform consists of several integrated subsystems: pressure control, temperature management, data acquisition, and safety monitoring. The Python control software interfaces with National Instruments hardware for precise control and measurement.

        The system employs a closed-loop pressure control algorithm that maintains target pressures within ±0.1% accuracy. Temperature is controlled using both heating elements and cooling circuits to achieve the required thermal cycling profiles.`,
        codePreview: {
          title: "Pressure Control Algorithm",
          preview: `class PressureController:
    def __init__(self, target_pressure, tolerance=0.001):
        self.target = target_pressure
        self.tolerance = tolerance
        self.pid = PIDController(kp=0.5, ki=0.1, kd=0.01)
    
    def control_loop(self):
        while self.running:
            current_pressure = self.read_pressure()
            error = self.target - current_pressure
            
            if abs(error) > self.tolerance:
                control_signal = self.pid.compute(error)
                self.adjust_pressure(control_signal)
            
            time.sleep(0.01)  # 100 Hz control loop`,
          fullCode: `import time
import numpy as np
from ni_control import NIDAQInterface
from pid_controller import PIDController

class PressureController:
    def __init__(self, target_pressure, tolerance=0.001):
        self.target = target_pressure
        self.tolerance = tolerance
        self.pid = PIDController(kp=0.5, ki=0.1, kd=0.01)
        self.daq = NIDAQInterface()
        self.running = False
        self.pressure_history = []
        
    def read_pressure(self):
        """Read current pressure from transducer"""
        voltage = self.daq.read_analog_channel('ai0')
        # Convert voltage to pressure (0-10V = 0-6000 psi)
        pressure = (voltage / 10.0) * 6000
        self.pressure_history.append(pressure)
        return pressure
        
    def adjust_pressure(self, control_signal):
        """Adjust pressure using proportional valve"""
        # Clamp control signal to safe limits
        control_signal = np.clip(control_signal, -10, 10)
        self.daq.write_analog_channel('ao0', control_signal)
        
    def control_loop(self):
        """Main pressure control loop"""
        self.running = True
        while self.running:
            current_pressure = self.read_pressure()
            error = self.target - current_pressure
            
            if abs(error) > self.tolerance:
                control_signal = self.pid.compute(error)
                self.adjust_pressure(control_signal)
                
            # Safety check
            if current_pressure > 6500:  # Emergency shutdown
                self.emergency_shutdown()
                break
                
            time.sleep(0.01)  # 100 Hz control loop
            
    def emergency_shutdown(self):
        """Emergency pressure release"""
        self.daq.write_digital_channel('port0/line0', True)  # Open relief valve
        self.running = False
        print("EMERGENCY SHUTDOWN - Overpressure detected!")`,
          language: "python"
        }
      },
      {
        type: "text-right",
        title: "Theoretical Background",
        content: `The heat transfer during Mars entry follows complex physics involving convective heating, radiation, and material ablation. The Fay-Riddell correlation provides the foundation for stagnation point heating calculations, while the Chapman-Rubesin parameter accounts for variable properties in the boundary layer.`,
        equations: [
          {
            equation: "q̇ = ρ∞U∞³√(ρw/ρ∞) × Pr^(-2/3)",
            variables: [
              { symbol: "q̇", description: "convective heat flux (W/cm²)" },
              { symbol: "ρ∞", description: "freestream density (kg/m³)" },
              { symbol: "U∞", description: "freestream velocity (m/s)" },
              { symbol: "Pr", description: "Prandtl number" }
            ]
          },
          {
            equation: "\\frac{∂T}{∂t} = α∇²T + \\frac{Q̇}{ρcp}",
            variables: [
              { symbol: "α", description: "thermal diffusivity (m²/s)" },
              { symbol: "Q̇", description: "heat generation rate (W/m³)" },
              { symbol: "cp", description: "specific heat capacity (J/kg·K)" }
            ]
          }
        ],
        codePreview: {
          title: "Fay-Riddell Stagnation Point Heating",
          preview: `import numpy as np

def fay_riddell_heating(rho_inf, u_inf, r_n, pr=0.71):
    """Calculate stagnation point heating rate"""
    C = 0.94  # Fay-Riddell constant
    q_dot = C * np.sqrt(rho_inf/r_n) * u_inf**3 * pr**(-2/3)
    return q_dot

# Mars entry conditions
rho_mars = 0.020  # kg/m³
velocity = 6200   # m/s
nose_radius = 2.25  # m

heating_rate = fay_riddell_heating(rho_mars, velocity, nose_radius)
print(f"Peak heating: {heating_rate:.2f} W/cm²")`,
          fullCode: `import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def fay_riddell_heating(rho_inf, u_inf, r_n, pr=0.71):
    """
    Calculate stagnation point heating rate using Fay-Riddell correlation
    
    Parameters:
    rho_inf: freestream density (kg/m³)
    u_inf: freestream velocity (m/s)
    r_n: nose radius (m)
    pr: Prandtl number
    
    Returns:
    q_dot: heating rate (W/cm²)
    """
    C = 0.94  # Fay-Riddell constant
    q_dot = C * np.sqrt(rho_inf/r_n) * u_inf**3 * pr**(-2/3)
    return q_dot

def heat_conduction_1d(T, t, alpha, thickness, q_surface):
    """
    1D heat conduction equation with surface heating
    """
    dTdt = np.zeros_like(T)
    dx = thickness / (len(T) - 1)
    
    # Interior points
    for i in range(1, len(T)-1):
        dTdt[i] = alpha * (T[i+1] - 2*T[i] + T[i-1]) / dx**2
    
    # Boundary conditions
    dTdt[0] = q_surface / (rho * cp * dx)  # Surface heating
    dTdt[-1] = 0  # Insulated back face
    
    return dTdt

# Mars entry trajectory data
time = np.linspace(0, 420, 1000)  # Entry duration: 420 seconds
altitude = 125000 - 3500 * time  # Linear approximation
velocity = 6200 * np.exp(-time/200)  # Velocity decay
rho_atm = 0.020 * np.exp((125000 - altitude)/10000)  # Exponential atmosphere

# Material properties (PICA-X)
rho = 280  # kg/m³
cp = 1500  # J/kg·K
k = 0.15   # W/m·K
alpha = k / (rho * cp)
thickness = 0.058  # m

# Calculate heating profile
nose_radius = 2.25  # m
heating_profile = []

for i, t in enumerate(time):
    q_dot = fay_riddell_heating(rho_atm[i], velocity[i], nose_radius)
    heating_profile.append(q_dot)

heating_profile = np.array(heating_profile)

# Solve heat conduction
n_nodes = 100
x = np.linspace(0, thickness, n_nodes)
T_initial = np.ones(n_nodes) * 300  # Initial temperature: 300 K

# Temperature evolution
T_solution = []
for i, q_surf in enumerate(heating_profile[::10]):  # Sample every 10 points
    if i == 0:
        T = T_initial
    else:
        t_span = [0, 4.2]  # 4.2 second intervals
        sol = odeint(heat_conduction_1d, T, t_span, args=(alpha, thickness, q_surf))
        T = sol[-1]
    T_solution.append(T.copy())

# Results analysis
max_surface_temp = max([T[0] for T in T_solution])
max_heating = max(heating_profile)

print(f"Maximum heating rate: {max_heating:.2f} W/cm²")
print(f"Maximum surface temperature: {max_surface_temp:.0f} K")
print(f"Peak temperature location: Surface")

# Create visualization
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Heating profile
ax1.plot(time, heating_profile, 'r-', linewidth=2)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Heat Flux (W/cm²)')
ax1.set_title('Convective Heating During Mars Entry')
ax1.grid(True, alpha=0.3)

# Velocity and altitude
ax2_twin = ax2.twinx()
ax2.plot(time, velocity/1000, 'b-', linewidth=2, label='Velocity')
ax2_twin.plot(time, altitude/1000, 'g-', linewidth=2, label='Altitude')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Velocity (km/s)', color='b')
ax2_twin.set_ylabel('Altitude (km)', color='g')
ax2.set_title('Entry Trajectory')
ax2.grid(True, alpha=0.3)

# Temperature distribution
time_samples = np.linspace(0, len(T_solution)-1, 5, dtype=int)
for i, t_idx in enumerate(time_samples):
    ax3.plot(x*1000, T_solution[t_idx], label=f't = {t_idx*42:.0f} s')
ax3.set_xlabel('Distance from surface (mm)')
ax3.set_ylabel('Temperature (K)')
ax3.set_title('Temperature Distribution Through Heat Shield')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Safety factor analysis
T_limit = 3000  # Material temperature limit (K)
safety_factor = T_limit / max_surface_temp
print(f"\\nSafety Analysis:")
print(f"Material temperature limit: {T_limit} K")
print(f"Maximum predicted temperature: {max_surface_temp:.0f} K")
print(f"Safety factor: {safety_factor:.2f}")

if safety_factor > 1.5:
    print("✓ Heat shield design meets safety requirements")
else:
    print("⚠ Heat shield design requires optimization")`
        }
      }
    ]
  },
  "rga-sensor-integration": {
    id: "rga-sensor-integration",
    title: "RGA Sensor Integration with Unitree Go2 Robot",
    subtitle: "Designed and simulated a robust mounting system to integrate a Residual Gas Analyzer sensor onto a quadruped robot",
    category: "Mechanical Design",
    date: "2024",
    author: "Engineering Team",
    tags: ["CAD Design", "Vibration Isolation", "Robotics Integration"],
    hero: "/lovable-uploads/7e9814d1-b051-4b58-99a9-b57a50fe4738.png",
    sections: [
      {
        type: "text-left",
        title: "Project Overview",
        content: `This project focused on developing a robust mounting system to integrate a Residual Gas Analyzer (RGA) sensor onto the Unitree Go2 quadruped robot. The challenge was to create a vibration-isolated mounting system that would protect the sensitive analytical instrument while maintaining the robot's mobility and balance.

        The RGA sensor requires precise environmental control and vibration isolation to function accurately. The mounting system needed to accommodate the sensor's weight distribution, provide adequate damping, and maintain accessibility for maintenance while ensuring the robot's dynamic stability during operation.`,
        visual: {
          type: "terminal",
          content: `// Robot Specifications
Robot Model: Unitree Go2
Payload Capacity: 3 kg
Operating Speed: 0.5 m/s
Step Height: 15 cm
Mass: 15 kg

// RGA Sensor Specifications  
Mass: 2.1 kg
Dimensions: 200×150×100 mm
Operating Temperature: 15-35°C
Vibration Sensitivity: <0.1g RMS
Power Consumption: 45W`
        }
      },
      {
        type: "text-right",
        title: "Mounting System Design",
        content: `The mounting system employs a three-stage vibration isolation approach: primary structural mounting, secondary damping layer, and tertiary fine-tuning isolators. The design utilizes finite element analysis to optimize the mounting geometry for minimal vibration transmission.

        The system features adjustable damping characteristics to accommodate different operating conditions and terrain types. Custom brackets distribute the sensor weight evenly across the robot's frame, maintaining the center of gravity within acceptable limits for stable locomotion.`,
        visual: {
          type: "terminal",
          content: `// Vibration Isolation Performance
Primary Stage: -20 dB @ 10-50 Hz
Secondary Stage: -15 dB @ 5-100 Hz  
Tertiary Stage: -10 dB @ 1-200 Hz
Total Isolation: -35 dB typical

// Material Properties
Bracket: Aluminum 6061-T6
Dampers: Sorbothane 50 Shore A
Isolators: Silicone gel 30 Shore A
Fasteners: 316 Stainless Steel`
        }
      }
    ]
  },
  "uav-propulsion-optimization": {
    id: "uav-propulsion-optimization",
    title: "UAV Propulsion Optimization via High-Fidelity Simulation",
    subtitle: "Conducted advanced CFD, combustion, and acoustic simulations of UAV propulsion systems for performance optimization",
    category: "CFD Analysis",
    date: "2024",
    author: "Engineering Team",
    tags: ["ANSYS Fluent", "CFD Modeling", "LMS Virtual.Lab"],
    hero: "/lovable-uploads/8cf36141-768e-42d1-9dd6-1da18d8ddee5.png",
    sections: [
      {
        type: "text-left",
        title: "Project Overview",
        content: `This project developed advanced CFD, combustion, and acoustic simulation capabilities for UAV propulsion system optimization using ANSYS Fluent and LMS Virtual.Lab. The study focused on multi-physics analysis to optimize propeller design, engine performance, and noise reduction for various UAV applications.

        The simulation framework integrates aerodynamic performance modeling with combustion analysis and acoustics prediction to provide comprehensive design insights. This approach enables simultaneous optimization of thrust efficiency, fuel consumption, and noise signature for different mission profiles.`,
        visual: {
          type: "terminal", 
          content: `// Simulation Parameters
CFD Mesh: 2.5M cells
Turbulence Model: k-ω SST
Combustion Model: PDF/FlameLet
Acoustic Analysis: FW-H equation
Convergence: 1e-5 residuals

// Performance Targets
Thrust Efficiency: >85%
Fuel Consumption: <0.3 kg/hr
Noise Level: <65 dB @ 100m
Operating Range: 0-4000m altitude`
        }
      }
    ]
  },
  "vibration-fatigue-detection": {
    id: "vibration-fatigue-detection",
    title: "Vibration-Based Fatigue Risk Detection for NASA's MSolo Mass Spectrometer",
    subtitle: "Developed real-time anomaly detection algorithms using FFT analysis and machine learning for fatigue risk assessment",
    category: "Signal Processing",
    date: "2024",
    author: "Engineering Team",
    tags: ["FFT Analysis", "Machine Learning", "Real-Time Detection"],
    hero: "/lovable-uploads/d1e74099-500d-4c46-a984-3fbe6f55a551.png",
    sections: [
      {
        type: "text-left",
        title: "Project Overview", 
        content: `This project focused on developing real-time anomaly detection algorithms for NASA's MSolo Mass Spectrometer using FFT analysis and machine learning techniques. The system monitors vibration signatures to detect early signs of fatigue and mechanical failure in the sensitive analytical instrument.

        The challenge was to create a robust detection system that could distinguish between normal operational vibrations and potentially damaging anomalous patterns. The solution employs advanced signal processing and machine learning algorithms to provide real-time fatigue risk assessment with minimal false positives.`,
        visual: {
          type: "terminal",
          content: `// System Specifications
Sampling Rate: 10 kHz
FFT Window: 2048 points
Frequency Range: 0-5000 Hz
Detection Latency: <100 ms
Accuracy: >95%

// Hardware Interface
Accelerometer: 3-axis MEMS
ADC Resolution: 16-bit
Data Interface: SPI
Power Consumption: 12W
Operating Range: -40°C to +85°C`
        }
      },
      {
        type: "text-right",
        title: "Signal Processing & Machine Learning",
        content: `The system employs a multi-stage analysis approach combining traditional FFT-based frequency analysis with modern machine learning classification. The FFT analysis extracts spectral features while the ML algorithm identifies patterns indicative of fatigue development.

        A sliding window approach enables continuous monitoring while feature extraction algorithms identify key indicators such as peak frequency shifts, harmonic distortion, and spectral energy distribution changes. The machine learning model was trained on extensive historical data to recognize fatigue signatures.`,
        codePreview: {
          title: "Real-Time Anomaly Detection Algorithm",
          preview: `import numpy as np
from scipy.fft import fft, fftfreq
from sklearn.ensemble import IsolationForest

class VibratingFatigueDetector:
    def __init__(self, sample_rate=10000, window_size=2048):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.baseline_features = None
        
    def extract_features(self, signal):
        # FFT analysis
        fft_vals = fft(signal)
        freqs = fftfreq(len(signal), 1/self.sample_rate)
        magnitude = np.abs(fft_vals)
        
        # Feature extraction
        features = {
            'peak_freq': freqs[np.argmax(magnitude)],
            'spectral_centroid': np.sum(freqs * magnitude) / np.sum(magnitude),
            'spectral_spread': np.sqrt(np.sum(((freqs - self.spectral_centroid)**2) * magnitude) / np.sum(magnitude)),
            'total_power': np.sum(magnitude**2)
        }
        
        return np.array(list(features.values()))
        
    def detect_anomaly(self, signal):
        features = self.extract_features(signal)
        anomaly_score = self.anomaly_detector.decision_function([features])[0]
        is_anomaly = anomaly_score < -0.1
        
        return is_anomaly, anomaly_score`,
          fullCode: `import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy import signal as scipy_signal
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import time
import threading
from collections import deque

class VibratingFatigueDetector:
    """
    Real-time vibration-based fatigue detection system for NASA MSolo Mass Spectrometer
    """
    
    def __init__(self, sample_rate=10000, window_size=2048, overlap=0.5):
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.overlap = overlap
        self.hop_size = int(window_size * (1 - overlap))
        
        # Machine learning components
        self.anomaly_detector = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Signal processing parameters
        self.freq_bins = fftfreq(window_size, 1/sample_rate)
        self.freq_bins = self.freq_bins[:window_size//2]  # Only positive frequencies
        
        # Monitoring state
        self.signal_buffer = deque(maxlen=window_size*2)
        self.feature_history = deque(maxlen=100)
        self.anomaly_history = deque(maxlen=50)
        self.baseline_features = None
        
        # Thresholds and parameters
        self.anomaly_threshold = -0.1
        self.fatigue_risk_threshold = 0.7
        self.alert_cooldown = 5.0  # seconds
        self.last_alert_time = 0
        
        # Callbacks
        self.alert_callback = None
        self.data_callback = None
        
    def extract_features(self, signal_window):
        """
        Extract comprehensive feature set from vibration signal
        """
        # Ensure signal is properly windowed
        windowed_signal = signal_window * scipy_signal.windows.hann(len(signal_window))
        
        # FFT analysis
        fft_vals = fft(windowed_signal)
        magnitude = np.abs(fft_vals[:len(fft_vals)//2])
        magnitude = magnitude / len(signal_window)  # Normalize
        
        # Frequency domain features
        power_spectrum = magnitude ** 2
        total_power = np.sum(power_spectrum)
        
        # Spectral features
        spectral_centroid = np.sum(self.freq_bins * power_spectrum) / total_power if total_power > 0 else 0
        spectral_spread = np.sqrt(np.sum(((self.freq_bins - spectral_centroid)**2) * power_spectrum) / total_power) if total_power > 0 else 0
        spectral_rolloff = self.calculate_spectral_rolloff(power_spectrum, 0.85)
        spectral_flux = self.calculate_spectral_flux(magnitude) if len(self.feature_history) > 0 else 0
        
        # Peak analysis
        peaks, _ = scipy_signal.find_peaks(magnitude, height=np.max(magnitude)*0.1)
        dominant_freq = self.freq_bins[np.argmax(magnitude)] if len(magnitude) > 0 else 0
        num_peaks = len(peaks)
        
        # Harmonic analysis
        harmonic_ratio = self.calculate_harmonic_ratio(magnitude, dominant_freq)
        
        # Time domain features
        rms = np.sqrt(np.mean(signal_window**2))
        peak_value = np.max(np.abs(signal_window))
        crest_factor = peak_value / rms if rms > 0 else 0
        skewness = scipy_signal.moment(signal_window, moment=3)
        kurtosis = scipy_signal.moment(signal_window, moment=4)
        
        # Fatigue-specific indicators
        high_freq_energy = np.sum(power_spectrum[self.freq_bins > 1000]) / total_power if total_power > 0 else 0
        low_freq_energy = np.sum(power_spectrum[self.freq_bins < 100]) / total_power if total_power > 0 else 0
        
        features = np.array([
            # Spectral features
            spectral_centroid,
            spectral_spread, 
            spectral_rolloff,
            spectral_flux,
            
            # Peak features
            dominant_freq,
            num_peaks,
            harmonic_ratio,
            
            # Time domain features
            rms,
            crest_factor,
            skewness,
            kurtosis,
            
            # Energy distribution
            high_freq_energy,
            low_freq_energy,
            total_power
        ])
        
        return features
    
    def calculate_spectral_rolloff(self, power_spectrum, rolloff_percent=0.85):
        """Calculate frequency below which specified percentage of total energy is contained"""
        cumsum = np.cumsum(power_spectrum)
        total_energy = cumsum[-1]
        rolloff_idx = np.where(cumsum >= rolloff_percent * total_energy)[0]
        return self.freq_bins[rolloff_idx[0]] if len(rolloff_idx) > 0 else self.freq_bins[-1]
    
    def calculate_spectral_flux(self, magnitude):
        """Calculate spectral flux (rate of change in magnitude spectrum)"""
        if len(self.feature_history) == 0:
            return 0
        prev_magnitude = self.feature_history[-1]['raw_magnitude']
        flux = np.sum((magnitude - prev_magnitude)**2)
        return flux
    
    def calculate_harmonic_ratio(self, magnitude, fundamental_freq):
        """Calculate ratio of harmonic energy to total energy"""
        if fundamental_freq == 0:
            return 0
        
        harmonic_freqs = [2*fundamental_freq, 3*fundamental_freq, 4*fundamental_freq]
        harmonic_energy = 0
        
        for harm_freq in harmonic_freqs:
            if harm_freq < self.freq_bins[-1]:
                idx = np.argmin(np.abs(self.freq_bins - harm_freq))
                harmonic_energy += magnitude[idx]**2
        
        total_energy = np.sum(magnitude**2)
        return harmonic_energy / total_energy if total_energy > 0 else 0
    
    def train_baseline(self, training_signals):
        """
        Train the anomaly detection model on baseline (normal) operation data
        """
        print("Training baseline model...")
        training_features = []
        
        for signal in training_signals:
            # Extract features from overlapping windows
            for i in range(0, len(signal) - self.window_size, self.hop_size):
                window = signal[i:i + self.window_size]
                features = self.extract_features(window)
                training_features.append(features)
        
        training_features = np.array(training_features)
        
        # Fit scaler and anomaly detector
        self.scaler.fit(training_features)
        scaled_features = self.scaler.transform(training_features)
        self.anomaly_detector.fit(scaled_features)
        
        # Store baseline statistics
        self.baseline_features = {
            'mean': np.mean(training_features, axis=0),
            'std': np.std(training_features, axis=0),
            'percentiles': np.percentile(training_features, [5, 25, 50, 75, 95], axis=0)
        }
        
        self.is_trained = True
        print(f"Model trained on {len(training_features)} feature vectors")
    
    def process_signal_chunk(self, signal_chunk):
        """
        Process a chunk of incoming signal data
        """
        if not self.is_trained:
            return None, "Model not trained"
        
        # Add to buffer
        self.signal_buffer.extend(signal_chunk)
        
        results = []
        
        # Process all complete windows in buffer
        while len(self.signal_buffer) >= self.window_size:
            # Extract window
            window = np.array(list(self.signal_buffer)[:self.window_size])
            
            # Extract features
            features = self.extract_features(window)
            scaled_features = self.scaler.transform([features])
            
            # Anomaly detection
            anomaly_score = self.anomaly_detector.decision_function(scaled_features)[0]
            is_anomaly = anomaly_score < self.anomaly_threshold
            
            # Calculate fatigue risk
            fatigue_risk = self.calculate_fatigue_risk(features, anomaly_score)
            
            # Store results
            result = {
                'timestamp': time.time(),
                'features': features,
                'raw_magnitude': np.abs(fft(window))[:len(window)//2],
                'anomaly_score': anomaly_score,
                'is_anomaly': is_anomaly,
                'fatigue_risk': fatigue_risk,
                'alert_level': self.determine_alert_level(fatigue_risk, is_anomaly)
            }
            
            self.feature_history.append(result)
            self.anomaly_history.append(is_anomaly)
            results.append(result)
            
            # Check for alerts
            self.check_alerts(result)
            
            # Remove processed samples from buffer
            for _ in range(self.hop_size):
                if self.signal_buffer:
                    self.signal_buffer.popleft()
        
        return results
    
    def calculate_fatigue_risk(self, features, anomaly_score):
        """
        Calculate fatigue risk based on features and anomaly score
        """
        if self.baseline_features is None:
            return 0.0
        
        # Normalize features relative to baseline
        baseline_mean = self.baseline_features['mean']
        baseline_std = self.baseline_features['std']
        
        # Calculate deviations from baseline
        normalized_deviations = np.abs(features - baseline_mean) / (baseline_std + 1e-8)
        max_deviation = np.max(normalized_deviations)
        
        # Combine anomaly score and feature deviations
        anomaly_factor = max(0, -anomaly_score)  # Convert to positive scale
        deviation_factor = min(max_deviation / 3.0, 1.0)  # Normalize to 0-1
        
        # Historical anomaly frequency
        recent_anomaly_rate = np.mean(list(self.anomaly_history)) if self.anomaly_history else 0
        
        # Weighted combination
        fatigue_risk = (0.4 * anomaly_factor + 
                       0.4 * deviation_factor + 
                       0.2 * recent_anomaly_rate)
        
        return np.clip(fatigue_risk, 0.0, 1.0)
    
    def determine_alert_level(self, fatigue_risk, is_anomaly):
        """
        Determine alert level based on fatigue risk and anomaly detection
        """
        if fatigue_risk > 0.8 or is_anomaly:
            return "CRITICAL"
        elif fatigue_risk > 0.6:
            return "WARNING"
        elif fatigue_risk > 0.4:
            return "CAUTION"
        else:
            return "NORMAL"
    
    def check_alerts(self, result):
        """
        Check if alerts should be triggered
        """
        current_time = time.time()
        
        # Respect cooldown period
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        alert_level = result['alert_level']
        
        if alert_level in ["CRITICAL", "WARNING"] and self.alert_callback:
            self.alert_callback(result)
            self.last_alert_time = current_time
    
    def set_alert_callback(self, callback):
        """Set callback function for alerts"""
        self.alert_callback = callback
    
    def set_data_callback(self, callback):
        """Set callback function for data updates"""
        self.data_callback = callback
    
    def generate_report(self):
        """
        Generate comprehensive health report
        """
        if not self.feature_history:
            return "No data available for report generation"
        
        recent_data = list(self.feature_history)[-20:]  # Last 20 measurements
        
        # Calculate statistics
        fatigue_risks = [d['fatigue_risk'] for d in recent_data]
        anomaly_scores = [d['anomaly_score'] for d in recent_data]
        
        avg_fatigue_risk = np.mean(fatigue_risks)
        max_fatigue_risk = np.max(fatigue_risks)
        anomaly_rate = np.mean([d['is_anomaly'] for d in recent_data])
        
        # Trend analysis
        if len(fatigue_risks) >= 10:
            trend = np.polyfit(range(len(fatigue_risks)), fatigue_risks, 1)[0]
            trend_desc = "increasing" if trend > 0.01 else "decreasing" if trend < -0.01 else "stable"
        else:
            trend_desc = "insufficient data"
        
        report = f"""
        FATIGUE DETECTION SYSTEM REPORT
        ===============================
        
        Overall Health Status: {self.determine_overall_health(avg_fatigue_risk, anomaly_rate)}
        
        Recent Statistics:
        - Average Fatigue Risk: {avg_fatigue_risk:.3f}
        - Maximum Fatigue Risk: {max_fatigue_risk:.3f}
        - Anomaly Rate: {anomaly_rate:.1%}
        - Risk Trend: {trend_desc}
        
        Recommendations:
        {self.generate_recommendations(avg_fatigue_risk, anomaly_rate, trend_desc)}
        """
        
        return report
    
    def determine_overall_health(self, avg_risk, anomaly_rate):
        """Determine overall system health"""
        if avg_risk > 0.7 or anomaly_rate > 0.3:
            return "POOR - Immediate attention required"
        elif avg_risk > 0.5 or anomaly_rate > 0.2:
            return "FAIR - Monitor closely"
        elif avg_risk > 0.3 or anomaly_rate > 0.1:
            return "GOOD - Normal operation"
        else:
            return "EXCELLENT - Optimal condition"
    
    def generate_recommendations(self, avg_risk, anomaly_rate, trend):
        """Generate maintenance recommendations"""
        recommendations = []
        
        if avg_risk > 0.6:
            recommendations.append("- Schedule immediate inspection of mechanical components")
        
        if anomaly_rate > 0.2:
            recommendations.append("- Check for loose connections or mounting issues")
        
        if trend == "increasing":
            recommendations.append("- Monitor system more frequently")
            recommendations.append("- Consider preventive maintenance")
        
        if not recommendations:
            recommendations.append("- Continue normal operation")
            recommendations.append("- Maintain regular monitoring schedule")
        
        return "\\n".join(recommendations)
    
    def save_model(self, filepath):
        """Save trained model to file"""
        model_data = {
            'anomaly_detector': self.anomaly_detector,
            'scaler': self.scaler,
            'baseline_features': self.baseline_features,
            'parameters': {
                'sample_rate': self.sample_rate,
                'window_size': self.window_size,
                'overlap': self.overlap
            }
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model from file"""
        model_data = joblib.load(filepath)
        self.anomaly_detector = model_data['anomaly_detector']
        self.scaler = model_data['scaler']
        self.baseline_features = model_data['baseline_features']
        self.is_trained = True
        print(f"Model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize detector
    detector = VibratingFatigueDetector(sample_rate=10000, window_size=2048)
    
    # Define alert callback
    def alert_handler(result):
        print(f"ALERT: {result['alert_level']} - Fatigue Risk: {result['fatigue_risk']:.3f}")
        print(f"Anomaly Score: {result['anomaly_score']:.3f}")
        print(f"Timestamp: {time.ctime(result['timestamp'])}")
        print("-" * 50)
    
    detector.set_alert_callback(alert_handler)
    
    # Generate synthetic training data (normal operation)
    print("Generating training data...")
    training_signals = []
    for i in range(10):
        t = np.linspace(0, 10, 100000)  # 10 seconds at 10kHz
        # Normal operation: low amplitude, stable frequency
        normal_signal = (0.1 * np.sin(2*np.pi*50*t) + 
                        0.05 * np.sin(2*np.pi*120*t) + 
                        0.02 * np.random.randn(len(t)))
        training_signals.append(normal_signal)
    
    # Train baseline model
    detector.train_baseline(training_signals)
    
    # Simulate real-time operation
    print("\\nStarting real-time simulation...")
    
    # Normal operation
    print("Phase 1: Normal operation")
    for i in range(20):
        t = np.linspace(0, 0.5, 5000)  # 0.5 seconds at 10kHz
        normal_signal = (0.1 * np.sin(2*np.pi*50*t) + 
                        0.05 * np.sin(2*np.pi*120*t) + 
                        0.02 * np.random.randn(len(t)))
        
        results = detector.process_signal_chunk(normal_signal)
        if results:
            latest = results[-1]
            print(f"Normal operation - Risk: {latest['fatigue_risk']:.3f}, Status: {latest['alert_level']}")
        
        time.sleep(0.1)  # Simulate real-time delay
    
    # Introduce anomalies (fatigue development)
    print("\\nPhase 2: Developing fatigue conditions")
    for i in range(20):
        t = np.linspace(0, 0.5, 5000)
        # Gradual increase in amplitude and frequency content (fatigue signature)
        fatigue_factor = 1 + 0.1 * i  # Gradually increasing
        anomalous_signal = (fatigue_factor * 0.15 * np.sin(2*np.pi*50*t) + 
                           fatigue_factor * 0.08 * np.sin(2*np.pi*120*t) +
                           0.03 * np.sin(2*np.pi*300*t) +  # Higher frequency component
                           0.03 * np.random.randn(len(t)))
        
        results = detector.process_signal_chunk(anomalous_signal)
        if results:
            latest = results[-1]
            print(f"Fatigue development - Risk: {latest['fatigue_risk']:.3f}, Status: {latest['alert_level']}")
        
        time.sleep(0.1)
    
    # Generate final report
    print("\\n" + detector.generate_report())
    
    # Save model for future use
    detector.save_model('fatigue_detector_model.pkl')
    
    print("\\nSimulation completed successfully!")`,
          language: "python"
        }
      },
      {
        type: "text-right",
        title: "Multi-Physics Simulation Framework", 
        content: `The simulation approach combines Reynolds-Averaged Navier-Stokes (RANS) equations for flow field analysis with detailed combustion modeling using probability density function methods. Acoustic analysis employs the Ffowcs Williams-Hawkings equation to predict far-field noise characteristics.

        The framework utilizes high-performance computing clusters to enable parametric studies across multiple design variables. Automated mesh generation and adaptive refinement ensure accurate capture of critical flow phenomena including boundary layer separation, combustion instabilities, and acoustic wave propagation.`,
        codePreview: {
          title: "ANSYS Fluent Automation Script",
          preview: `# ANSYS Fluent Simulation Setup
import ansys.fluent.core as pyfluent
from ansys.fluent.core import launch_fluent

# Launch Fluent session
solver = launch_fluent(precision='double', processor_count=16)

# Setup physics models
solver.setup.models.viscous.k_omega_sst()
solver.setup.models.energy.enable()
solver.setup.models.species.enable()

# Define boundary conditions
solver.setup.boundary_conditions.velocity_inlet.create(
    zone_name='inlet',
    velocity_magnitude=50,  # m/s
    turbulent_intensity=0.05
)

# Run calculation
solver.solution.run_calculation.iterate(iter=2000)`,
          fullCode: `# ANSYS Fluent UAV Propulsion Simulation
import ansys.fluent.core as pyfluent
from ansys.fluent.core import launch_fluent
import numpy as np
import matplotlib.pyplot as plt

def setup_fluent_simulation(mesh_file, operating_conditions):
    """
    Setup ANSYS Fluent simulation for UAV propulsion analysis
    """
    # Launch Fluent with multiple processors
    solver = launch_fluent(
        precision='double', 
        processor_count=16,
        show_gui=False
    )
    
    # Read mesh
    solver.file.read_case(file_name=mesh_file)
    
    # Physics models setup
    solver.setup.models.viscous.k_omega_sst.enable()
    solver.setup.models.energy.enable()
    solver.setup.models.species.enable()
    
    # Combustion model
    solver.setup.models.species.pdf_transport.enable()
    solver.setup.models.species.flamelet.enable()
    
    # Material properties
    solver.setup.materials.fluid.air.density.ideal_gas()
    
    # Operating conditions
    solver.setup.general.operating_conditions.operating_pressure = operating_conditions['pressure']
    solver.setup.general.operating_conditions.gravity.vector = [0, 0, -9.81]
    
    return solver

def set_boundary_conditions(solver, flight_conditions):
    """
    Configure boundary conditions for UAV flight simulation
    """
    # Inlet conditions
    solver.setup.boundary_conditions.velocity_inlet.create(
        zone_name='air_inlet',
        velocity_magnitude=flight_conditions['airspeed'],
        temperature=flight_conditions['temperature'],
        turbulent_intensity=0.05,
        turbulent_viscosity_ratio=10
    )
    
    # Fuel inlet
    solver.setup.boundary_conditions.mass_flow_inlet.create(
        zone_name='fuel_inlet',
        mass_flow_rate=flight_conditions['fuel_flow'],
        temperature=flight_conditions['fuel_temp']
    )
    
    # Engine outlet
    solver.setup.boundary_conditions.pressure_outlet.create(
        zone_name='exhaust',
        pressure=flight_conditions['exhaust_pressure']
    )
    
    # Propeller surfaces
    solver.setup.boundary_conditions.wall.no_slip.create(
        zone_name='propeller_blades',
        wall_motion='moving_wall',
        rotational_speed=flight_conditions['rpm']
    )

def run_parametric_study(base_conditions, param_ranges):
    """
    Execute parametric study for propulsion optimization
    """
    results = []
    
    for rpm in param_ranges['rpm']:
        for fuel_flow in param_ranges['fuel_flow']:
            for pitch_angle in param_ranges['pitch']:
                
                # Update conditions
                conditions = base_conditions.copy()
                conditions['rpm'] = rpm
                conditions['fuel_flow'] = fuel_flow
                conditions['pitch_angle'] = pitch_angle
                
                # Setup and run simulation
                solver = setup_fluent_simulation('uav_mesh.cas', conditions)
                set_boundary_conditions(solver, conditions)
                
                # Solution methods
                solver.solution.methods.scheme.coupled()
                solver.solution.methods.gradient_scheme.least_squares_cell_based()
                solver.solution.methods.pressure_discretization.second_order()
                solver.solution.methods.momentum_discretization.second_order_upwind()
                
                # Initialize and iterate
                solver.solution.initialization.hybrid_initialize()
                solver.solution.run_calculation.iterate(iter=1500)
                
                # Extract results
                thrust = solver.solution.report_definitions.force.create(
                    zone_names=['propeller_blades'],
                    direction_vector=[1, 0, 0]
                )
                
                power = solver.solution.report_definitions.moment.create(
                    zone_names=['propeller_blades'],
                    moment_axis=[1, 0, 0]
                ) * rpm * 2 * np.pi / 60
                
                efficiency = thrust['force'] * conditions['airspeed'] / power['moment']
                
                # Acoustic analysis
                acoustic_data = calculate_acoustics(solver, conditions)
                
                results.append({
                    'rpm': rpm,
                    'fuel_flow': fuel_flow,
                    'pitch': pitch_angle,
                    'thrust': thrust['force'],
                    'power': power['moment'],
                    'efficiency': efficiency,
                    'noise_level': acoustic_data['spl_100m']
                })
                
                solver.exit()
    
    return results

def calculate_acoustics(solver, conditions):
    """
    Perform acoustic analysis using Ffowcs Williams-Hawkings
    """
    # Enable acoustics model
    solver.models.acoustics.ffowcs_williams_hawkings.enable()
    
    # Define acoustic surfaces
    solver.acoustics.ffowcs_williams_hawkings.surface.create(
        name='propeller_surface',
        zone_names=['propeller_blades']
    )
    
    # Set receiver locations
    receivers = []
    for angle in range(0, 360, 30):
        x = 100 * np.cos(np.radians(angle))  # 100m radius
        y = 100 * np.sin(np.radians(angle))
        z = 0
        receivers.append([x, y, z])
    
    solver.acoustics.ffowcs_williams_hawkings.receivers.create(
        coordinates=receivers
    )
    
    # Calculate acoustic field
    solver.acoustics.ffowcs_williams_hawkings.calculate()
    
    # Extract sound pressure levels
    spl_data = solver.acoustics.ffowcs_williams_hawkings.results.spl()
    
    return {
        'spl_100m': np.mean(spl_data),
        'directivity': spl_data,
        'frequency_spectrum': solver.acoustics.ffowcs_williams_hawkings.results.spectrum()
    }

# Main execution
if __name__ == "__main__":
    # Define flight conditions
    base_flight_conditions = {
        'airspeed': 50,      # m/s
        'altitude': 1000,    # m
        'temperature': 280,  # K
        'pressure': 89875,   # Pa
        'fuel_flow': 0.001,  # kg/s
        'fuel_temp': 298,    # K
        'exhaust_pressure': 85000,  # Pa
        'rpm': 2400,         # RPM
        'pitch_angle': 15    # degrees
    }
    
    # Parameter ranges for optimization
    param_ranges = {
        'rpm': [2000, 2200, 2400, 2600, 2800],
        'fuel_flow': [0.0008, 0.001, 0.0012, 0.0014],
        'pitch': [12, 15, 18, 21]
    }
    
    # Run parametric study
    print("Starting UAV propulsion optimization study...")
    optimization_results = run_parametric_study(base_flight_conditions, param_ranges)
    
    # Analyze results
    max_efficiency = max(optimization_results, key=lambda x: x['efficiency'])
    min_noise = min(optimization_results, key=lambda x: x['noise_level'])
    
    print(f"Maximum efficiency: {max_efficiency['efficiency']:.3f}")
    print(f"Optimal RPM: {max_efficiency['rpm']}")
    print(f"Minimum noise level: {min_noise['noise_level']:.1f} dB")
    
    # Generate performance maps
    plt.figure(figsize=(15, 5))
    
    # Efficiency map
    plt.subplot(1, 3, 1)
    rpm_vals = [r['rpm'] for r in optimization_results]
    eff_vals = [r['efficiency'] for r in optimization_results]
    plt.scatter(rpm_vals, eff_vals, c='blue', alpha=0.6)
    plt.xlabel('RPM')
    plt.ylabel('Propulsive Efficiency')
    plt.title('Efficiency vs RPM')
    plt.grid(True)
    
    # Power vs Thrust
    plt.subplot(1, 3, 2)
    power_vals = [r['power'] for r in optimization_results]
    thrust_vals = [r['thrust'] for r in optimization_results]
    plt.scatter(power_vals, thrust_vals, c='red', alpha=0.6)
    plt.xlabel('Power (W)')
    plt.ylabel('Thrust (N)')
    plt.title('Thrust vs Power')
    plt.grid(True)
    
    # Noise characteristics
    plt.subplot(1, 3, 3)
    noise_vals = [r['noise_level'] for r in optimization_results]
    plt.scatter(rpm_vals, noise_vals, c='green', alpha=0.6)
    plt.xlabel('RPM')
    plt.ylabel('Noise Level (dB)')
    plt.title('Noise vs RPM')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('uav_propulsion_optimization.png', dpi=300)
    plt.show()
    
    print("Optimization study completed successfully!")`,
          language: "python"
        }
      }
    ]
  },
  "uav-tail-fuselage": {
    id: "uav-tail-fuselage",
    title: "UAV Tail & Fuselage Variations for Stability Analysis",
    subtitle: "Performed comprehensive stability analysis of various UAV design configurations to optimize flight performance",
    category: "Aerodynamics",
    date: "2024",
    author: "Engineering Team",
    tags: ["Stability Analysis", "Flight Dynamics", "Design Optimization"],
    hero: "/lovable-uploads/000f98ca-15f2-4d60-a820-a33b989ababe.png",
    sections: [
      {
        type: "text-left",
        title: "Project Overview",
        content: `This project performed comprehensive stability analysis of various UAV tail and fuselage design configurations to optimize flight performance and control characteristics. The study examined multiple geometric variations and their impact on longitudinal and lateral-directional stability margins.

        The analysis employed computational fluid dynamics and classical aerodynamic theory to evaluate stability derivatives and control authority for different design configurations. Wind tunnel testing validated computational results and provided experimental data for model refinement and optimization recommendations.`,
        visual: {
          type: "terminal",
          content: `// Design Configurations Tested
Tail Configurations: 5 variants
Fuselage Lengths: 3 options  
Wing Positions: High, Mid, Low
Control Surface Areas: Variable
Test Conditions: M = 0.1-0.7

// Stability Criteria
Static Margin: >5% MAC
Dutch Roll Damping: >0.1
Spiral Mode: Stable
Phugoid Damping: >0.04
Roll Mode: τ < 1.0 sec`
        }
      },
      {
        type: "text-right",
        title: "Stability Analysis & Results",
        content: `The stability analysis utilized both computational methods and wind tunnel testing to determine aerodynamic derivatives and assess flight characteristics. Key stability parameters including static margin, Dutch roll damping, and control power were evaluated across the flight envelope.

        Results showed that configuration optimization could improve stability margins by up to 25% while maintaining adequate control authority. The high-wing configuration with extended fuselage provided optimal balance between stability and maneuverability for the intended mission profile.`,
        visual: {
          type: "terminal",
          content: `// Optimized Configuration Results
Static Margin: 8.5% MAC (Target: >5%)
Dutch Roll Frequency: 1.2 rad/s
Dutch Roll Damping: 0.15 (Target: >0.1)
Roll Time Constant: 0.8 sec (Target: <1.0)
Spiral Mode: Stable divergence
Control Power: 15 deg/s² roll rate

// Performance Improvements
Stability Margin: +25%
Control Authority: +15%  
Gust Response: -20%
Pilot Workload: Reduced`
        }
      }
    ]
  }
};

const ProjectDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const project = projectsData[id as keyof typeof projectsData];

  if (!project) {
    return (
      <div className="min-h-screen bg-background">
        <Navigation />
        <div className="container mx-auto px-4 pt-24 text-center">
          <h1 className="text-4xl font-bold mb-4">Project Not Found</h1>
          <Button onClick={() => navigate('/projects')}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Projects
          </Button>
        </div>
      </div>
    );
  }

  const renderSection = (section: any, index: number) => {
    const isEven = index % 2 === 0;
    const textFirst = section.type === "text-left" ? isEven : !isEven;

    return (
      <section 
        key={index} 
        className={`py-16 animate-fade-in`}
        style={{ animationDelay: `${index * 0.2}s` }}
      >
        <div className="container mx-auto px-4">
          <div className={`grid lg:grid-cols-2 gap-12 items-start ${textFirst ? '' : 'lg:grid-flow-col-dense'}`}>
            {/* Text Content */}
            <div className={`space-y-6 ${textFirst ? '' : 'lg:col-start-2'}`}>
              <div className="engineering-callout">
                <h2 className="text-3xl font-bold mb-6 text-primary">{section.title}</h2>
                <div className="prose prose-lg max-w-none text-foreground/90 leading-relaxed">
                  {section.content.split('\n\n').map((paragraph: string, pIndex: number) => (
                    <p key={pIndex} className="mb-4">
                      {paragraph}
                    </p>
                  ))}
                </div>
              </div>

              {/* Mathematical Equations */}
              {section.equations && section.equations.map((eq: any, eqIndex: number) => (
                <MathEquation
                  key={eqIndex}
                  equation={eq.equation}
                  variables={eq.variables}
                />
              ))}

              {/* Engineering Standards Callout */}
              {section.standards && (
                <div className="engineering-callout">
                  <h4 className="font-semibold mb-2">Engineering Standards</h4>
                  <ul className="space-y-1 text-sm">
                    {section.standards.map((standard: string, sIndex: number) => (
                      <li key={sIndex} className="flex items-center">
                        <ChevronRight className="w-4 h-4 mr-2 text-accent" />
                        <span className="metrics-badge">{standard}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Pull Quote */}
              {section.pullQuote && (
                <blockquote className="pull-quote">
                  "{section.pullQuote}"
                </blockquote>
              )}

              {/* Metrics */}
              {section.metrics && (
                <div className="flex flex-wrap gap-2">
                  {section.metrics.map((metric: any, mIndex: number) => (
                    <span key={mIndex} className="metrics-badge">
                      {metric.label}: {metric.value}
                    </span>
                  ))}
                </div>
              )}
            </div>

            {/* Visual/Code Content */}
            <div className={`${textFirst ? 'lg:col-start-2' : ''}`}>
              {section.visual && section.visual.type === "terminal" && (
                <div className="bg-[#0D1117] rounded-lg border border-gray-800 overflow-hidden shadow-lg">
                  <div className="flex items-center px-4 py-2 bg-[#21262d] border-b border-gray-800">
                    <div className="flex space-x-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                    </div>
                    <span className="ml-4 text-sm text-gray-400">Terminal</span>
                  </div>
                  <div className="p-4">
                    <pre className="text-sm text-gray-100 font-mono whitespace-pre-wrap">
                      {section.visual.content}
                    </pre>
                  </div>
                </div>
              )}

              {section.codePreview && (
                <CodePreview
                  title={section.codePreview.title}
                  preview={section.codePreview.preview}
                  fullCode={section.codePreview.fullCode}
                />
              )}

              {section.image && (
                <div className="rounded-lg overflow-hidden shadow-lg">
                  <img 
                    src={section.image} 
                    alt={section.title}
                    className="w-full h-auto"
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    );
  };

  return (
    <div className="min-h-screen bg-background">
      <Navigation />
      
      {/* Hero Section */}
      <section className="relative pt-24 pb-16 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-background to-accent/10" />
        <div className="container mx-auto px-4 relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            <div className="flex items-center justify-center gap-4 mb-6">
              <Badge variant="secondary" className="text-sm">{project.category}</Badge>
              <div className="flex items-center text-sm text-muted-foreground">
                <Calendar className="w-4 h-4 mr-2" />
                {project.date}
              </div>
              <div className="flex items-center text-sm text-muted-foreground">
                <User className="w-4 h-4 mr-2" />
                {project.author}
              </div>
            </div>
            
            <h1 className="text-5xl font-bold mb-4 text-gradient">{project.title}</h1>
            <p className="text-xl text-muted-foreground mb-8 max-w-3xl mx-auto">
              {project.subtitle}
            </p>
            
            <div className="flex flex-wrap justify-center gap-2 mb-8">
              {project.tags.map((tag: string, index: number) => (
                <Badge key={index} variant="outline" className="text-sm">
                  {tag}
                </Badge>
              ))}
            </div>

            <Button 
              variant="outline" 
              onClick={() => navigate('/projects')}
              className="inline-flex items-center"
            >
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Projects
            </Button>
          </div>
        </div>
      </section>

      {/* Project Sections */}
      <div className="relative">
        {project.sections.map((section: any, index: number) => renderSection(section, index))}
      </div>

      {/* Back to Projects */}
      <section className="py-16 border-t border-border/50">
        <div className="container mx-auto px-4 text-center">
          <Button 
            onClick={() => navigate('/projects')}
            className="btn-primary"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Projects
          </Button>
        </div>
      </section>

      {/* Contact Section */}
      <ContactSection />
      
      {/* Footer */}
      <Footer />
    </div>
  );
};

export default ProjectDetail;
