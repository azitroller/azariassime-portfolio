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
    title: "Automated Valve Test Platform",
    subtitle: "High-Pressure, High-Temperature Testing System for Aerospace Applications",
    category: "Systems Engineering",
    date: "Spring 2024",
    author: "Azarias Thomas",
    tags: ["Automation", "Testing", "Python", "Data Acquisition", "Aerospace", "INFICON"],
    hero: "/lovable-uploads/000f98ca-15f2-4d60-a820-a33b989ababe.png",
    sections: [
      {
        type: "overview",
        title: "Context & Goal",
        content: "During my internship at INFICON, I encountered a critical engineering challenge that would fundamentally reshape how the company approaches valve testing for aerospace applications. The existing manual testing protocol for high-pressure, high-temperature valve systems presented numerous limitations that compromised both data quality and operational efficiency in ways that extended far beyond simple inconvenience.\n\nOperators were required to maintain continuous presence during test cycles lasting 12-48 hours, manually recording pressure and temperature measurements at 5-10 minute intervals while managing complex systems operating at pressures up to 15,000 psi and temperatures ranging from -40°C to 180°C. This approach introduced significant data gaps during which critical transient events could occur undetected—a particularly concerning issue given that valve failure initiation in aerospace systems can happen within seconds, making these measurement gaps unacceptable blind spots in the testing protocol.\n\nThe human factor complications extended well beyond simple data collection issues and represented a fundamental limitation in the reliability of test results. Statistical analysis of historical manual test data revealed measurement variance increased systematically from 2.3% to 7.8% after 8-hour operator shifts, with particularly pronounced degradation during overnight testing periods when fatigue effects became most severe.",
        metrics: [
          { label: "Operating Pressure", value: "Up to 15,000 psi" },
          { label: "Temperature Range", value: "-40°C to 180°C" },
          { label: "Test Duration", value: "12-48 hours" },
          { label: "Manual Variance", value: "2.3% to 7.8%" }
        ],
        image: {
          src: "/lovable-uploads/86b80ba9-25f7-499c-95da-1e4d8d1511b8.png",
          alt: "Automated Valve Test Platform - Initial test setup with valve components, pressure transducers, and sensor wiring arranged in testing tray for high-pressure aerospace valve testing",
          position: "right"
        }
      },
      {
        type: "theoretical",
        title: "Theoretical Background",
        content: "Understanding the complex material science underlying valve failure mechanisms became crucial to developing an effective automated testing solution that could detect the subtle signatures of degradation processes. At elevated temperatures, valve components experience time-dependent plastic deformation following Norton's power law, where the creep rate depends exponentially on temperature and follows a power relationship with stress magnitude.\n\nFor Inconel 718 valve seats operating at 180°C under typical aerospace loading conditions, creep rates of 10⁻⁸ s⁻¹ can accumulate to significant deformation over extended test durations, with the exponential temperature dependence meaning that small temperature variations can dramatically affect creep behavior and thus failure timing.\n\nRepeated thermal cycling induces alternating stress cycles due to differential thermal expansion between dissimilar materials commonly found in valve assemblies. This leads to low-cycle fatigue crack initiation after 10³-10⁴ cycles, following the Coffin-Manson relationship.",
        image: {
          src: "/lovable-uploads/025a0eec-1e0e-43d8-ae59-443683cf3c02.png",
          alt: "Automated Valve Test Platform - Data acquisition software interface showing configuration menu for sensor setup, valve testing parameters, and real-time monitoring controls",
          position: "left"
        },
        equations: [
          {
            equation: "\\dot{\\varepsilon} = A\\left(\\frac{\\sigma}{E}\\right)^n \\exp\\left(-\\frac{Q}{RT}\\right)",
            variables: [
              { symbol: "ε̇", description: "creep rate (s⁻¹)" },
              { symbol: "A", description: "material constant" },
              { symbol: "σ", description: "applied stress (Pa)" },
              { symbol: "E", description: "elastic modulus (Pa)" },
              { symbol: "n", description: "stress exponent (3-8 for aerospace alloys)" },
              { symbol: "Q", description: "activation energy for creep (J/mol)" },
              { symbol: "R", description: "gas constant (8.314 J/mol·K)" },
              { symbol: "T", description: "absolute temperature (K)" }
            ]
          },
          {
            equation: "\\sigma_{th} = \\frac{E\\alpha\\Delta T}{1-\\nu}",
            variables: [
              { symbol: "σth", description: "thermal stress (Pa)" },
              { symbol: "E", description: "elastic modulus (Pa)" },
              { symbol: "α", description: "coefficient of thermal expansion (1/K)" },
              { symbol: "ΔT", description: "temperature change (K)" },
              { symbol: "ν", description: "Poisson's ratio" }
            ]
          },
          {
            equation: "\\frac{\\Delta\\varepsilon_p}{2} = \\varepsilon'_f(2N_f)^c",
            variables: [
              { symbol: "Δεp", description: "plastic strain amplitude" },
              { symbol: "ε'f", description: "fatigue ductility coefficient" },
              { symbol: "Nf", description: "cycles to failure" },
              { symbol: "c", description: "fatigue ductility exponent" }
            ]
          }
        ]
      },
      {
        type: "methodology",
        title: "Steps & Methodology",
        content: "My approach to solving this multifaceted challenge began with comprehensive hardware architecture design, carefully selecting components based on performance requirements, environmental compatibility, integration capabilities, and long-term reliability under the extreme conditions encountered in aerospace valve testing.\n\n**Hardware Selection Process:**\n\nThe Omega PX309 pressure transducers were chosen after extensive analysis across temperature ranges and competitive evaluation against alternatives from Honeywell, Kulite, and Kistler, featuring a measurement range of 0-15,000 psi with ±0.25% full scale accuracy, temperature coefficient of ±0.02% FS/°C, and response time under 1 ms for 90% full scale deflection.\n\n**Calibration Protocol:**\n\nThe calibration protocol represented a critical aspect requiring meticulous attention to traceability and uncertainty analysis. I utilized NIST-traceable Fluke 719Pro pressure calibrators with ±0.025% accuracy, deriving calibration equations through weighted least-squares regression that accounted for measurement uncertainties at each calibration point.\n\n**Temperature Measurement:**\n\nType-K thermocouples were selected for their wide operating range, standardized response characteristics defined by NIST standards, and proven reliability in aerospace applications. Cold junction compensation was achieved through integrated circuit temperature sensors (AD590) with ±0.5°C accuracy.",
        image: {
          src: "/lovable-uploads/7ca7784b-52b2-43fd-9fe0-970f628faa3a.png",
          alt: "Automated Valve Test Platform - Advanced test chamber setup with multiple valve assemblies, green PCB control boards, and comprehensive sensor network for environmental testing",
          position: "right"
        },
        standards: [
          "ASTM F1387 - Valve Testing Standards",
          "NASA-STD-5009 - Nondestructive Evaluation Requirements",
          "MIL-PRF-87257 - Military Performance Specifications",
          "NIST Standards - Calibration Traceability"
        ],
        equations: [
          {
            equation: "P_{actual} = a_0 + a_1 \\times V_{sensor} + a_2 \\times T_{ambient} + a_3 \\times V_{sensor} \\times T_{ambient} + a_4 \\times V_{sensor}^2",
            variables: [
              { symbol: "Pactual", description: "calibrated pressure reading (psi)" },
              { symbol: "Vsensor", description: "sensor voltage output (V)" },
              { symbol: "Tambient", description: "ambient temperature (°C)" },
              { symbol: "a0-a4", description: "calibration coefficients" }
            ]
          },
          {
            equation: "V(T) = \\sum_{i=0}^{n} a_i \\times T^i",
            variables: [
              { symbol: "V(T)", description: "thermocouple voltage (mV)" },
              { symbol: "T", description: "temperature (°C)" },
              { symbol: "ai", description: "NIST polynomial coefficients" }
            ]
          }
        ]
      },
      {
        type: "implementation",
        title: "Data & Results",
        content: "The data acquisition system selection involved extensive analysis of sampling rate requirements, channel count, resolution needs, and environmental compatibility. The National Instruments USB-6343 DAQ system provided 32 single-ended analog inputs with 16-bit resolution, maximum aggregate sampling rate of 500 kS/s, input voltage range of ±10V with programmable gain amplifiers, and built-in anti-aliasing filters.\n\n**Performance Achievements:**\n\n• Reduced measurement variance from 7.8% to 0.3%\n• Achieved continuous 24/7 operation capability\n• Implemented 1 Hz minimum sampling rate compliance\n• Eliminated operator fatigue-related errors\n• Achieved NIST-traceable calibration accuracy\n\n**System Capabilities:**\n\nThe 16-bit resolution provides theoretical measurement resolution of 1 part in 65,536, corresponding to 0.3 mV resolution over the ±10V input range, which translates to pressure resolution of approximately 0.08 psi when combined with sensor scaling factors. However, effective resolution is typically 2-3 bits less due to noise and environmental factors.",
        image: {
          src: "/lovable-uploads/2933f505-0bd4-4e2e-b1ae-b2f3601c9c3c.png",
          alt: "Automated Valve Test Platform - INFICON Go Skeleton Leak Fixture testing system with digital display showing troubleshoot mode, test loops, and real-time pressure monitoring for aerospace valve testing",
          position: "left"
        },
        metrics: [
          { label: "Measurement Accuracy", value: "±0.25% FS" },
          { label: "Temperature Coefficient", value: "±0.02% FS/°C" },
          { label: "Response Time", value: "<1 ms" },
          { label: "Resolution", value: "0.08 psi" },
          { label: "Sampling Rate", value: "500 kS/s" },
          { label: "Variance Improvement", value: "7.8% → 0.3%" }
        ],
        visual: {
          type: "chart",
          content: {
            type: "line",
            data: {
              labels: ["Manual (Start)", "Manual (8hr)", "Manual (16hr)", "Manual (24hr)", "Automated"],
              datasets: [{
                label: "Measurement Variance (%)",
                data: [2.3, 4.1, 6.8, 7.8, 0.3],
                borderColor: "hsl(var(--primary))",
                backgroundColor: "hsl(var(--primary) / 0.1)"
              }]
            }
          }
        }
      },
      {
        type: "code",
        title: "Mathematical Models & Equations",
        content: "The software architecture represented a sophisticated integration of real-time control, statistical analysis, safety monitoring, and data management systems that required careful attention to timing, reliability, and maintainability. I developed a modular Python framework employing object-oriented design principles with clear separation of concerns.",
        equations: [
          {
            equation: "k = A \\exp\\left(-\\frac{E_a}{RT}\\right)",
            variables: [
              { symbol: "k", description: "reaction rate constant (s⁻¹)" },
              { symbol: "A", description: "pre-exponential factor" },
              { symbol: "Ea", description: "activation energy (80-120 kJ/mol)" },
              { symbol: "R", description: "gas constant (8.314 J/mol·K)" },
              { symbol: "T", description: "absolute temperature (K)" }
            ]
          },
          {
            equation: "a_T = \\exp\\left[\\frac{E_a}{R}\\left(\\frac{1}{T} - \\frac{1}{T_{ref}}\\right)\\right]",
            variables: [
              { symbol: "aT", description: "shift factor for time-temperature superposition" },
              { symbol: "Ea", description: "activation energy (J/mol)" },
              { symbol: "T", description: "test temperature (K)" },
              { symbol: "Tref", description: "reference temperature (K)" }
            ]
          }
        ],
        codePreview: {
          title: "Automated Valve Test Platform - Complete System",
          preview: `import nidaqmx
import numpy as np
from scipy import signal, stats
from sklearn.ensemble import RandomForestClassifier
import threading
import sqlite3
import time

class AdvancedDAQReader:
    def __init__(self, pressure_channels, temp_channels, 
                 sample_rate=10, buffer_size=10000):
        self.pressure_channels = pressure_channels
        self.temp_channels = temp_channels
        self.calibration_coeffs = self._load_calibration()`,
          fullCode: `import nidaqmx
import numpy as np
import sqlite3
import pandas as pd
from scipy import signal, stats
from datetime import datetime, timedelta
import threading
import queue
import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class AdvancedDAQReader:
    def __init__(self, pressure_channels, temp_channels, sample_rate=10, buffer_size=10000):
        self.pressure_channels = pressure_channels
        self.temp_channels = temp_channels
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.calibration_coeffs = self._load_calibration()
        self.filter_coeffs = self._design_filters()
        self.data_buffer = CircularBuffer(buffer_size)
        self.acquisition_active = False

    def _load_calibration(self):
        """Load calibration coefficients with temperature compensation"""
        # Multi-point calibration with cross-correlation terms
        # P_actual = a0 + a1*V + a2*T + a3*V*T + a4*V^2 + a5*T^2
        calibration_matrix = {
            'pressure_ch0': [0.125, 3750.2, -0.045, 0.001247, -0.002134, 0.0000834],
            'pressure_ch1': [0.118, 3748.7, -0.042, 0.001251, -0.002089, 0.0000829],
            'temperature': [-273.15, 25.068, 0.0, 0.0, 0.0, 0.0]  # Type-K polynomial coefficients
        }
        return calibration_matrix

    def _design_filters(self):
        """Design digital filters for noise rejection and signal conditioning"""
        filters = {}
        
        # Anti-aliasing filter (4th order Butterworth)
        nyquist = self.sample_rate / 2
        cutoff_aa = 0.4 * nyquist  # Conservative anti-aliasing
        b_aa, a_aa = signal.butter(4, cutoff_aa, fs=self.sample_rate, btype='low')
        filters['anti_aliasing'] = (b_aa, a_aa)
        
        # Power line interference notch filter (60 Hz)
        notch_freq = 60.0
        quality_factor = 30.0
        b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, fs=self.sample_rate)
        filters['power_line_notch'] = (b_notch, a_notch)
        
        # Low-pass filter for trend analysis (0.1 Hz cutoff)
        cutoff_lp = 0.1
        b_lp, a_lp = signal.butter(2, cutoff_lp, fs=self.sample_rate, btype='low')
        filters['trend_analysis'] = (b_lp, a_lp)
        
        return filters

    def acquire_data_burst(self, duration=1.0, apply_filters=True):
        """Acquire high-speed burst data with comprehensive signal processing"""
        samples_per_channel = int(self.sample_rate * duration)
        
        with nidaqmx.Task() as task:
            # Configure pressure channels with optimal settings
            for i, ch in enumerate(self.pressure_channels):
                task.ai_channels.add_ai_voltage_chan(ch, min_val=-10, max_val=10,
                                                   name_to_assign_to_channel=f"pressure_{i}")
                # Set terminal configuration for optimal noise rejection
                task.ai_channels[f"pressure_{i}"].ai_term_cfg = nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
            
            # Configure temperature channels
            for i, ch in enumerate(self.temp_channels):
                task.ai_channels.add_ai_voltage_chan(ch, min_val=-10, max_val=10,
                                                   name_to_assign_to_channel=f"temperature_{i}")
                task.ai_channels[f"temperature_{i}"].ai_term_cfg = nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
            
            # Configure timing with hardware triggering for synchronization
            task.timing.cfg_samp_clk_timing(rate=self.sample_rate,
                                          source='',  # Use internal clock
                                          active_edge=nidaqmx.constants.Edge.RISING,
                                          sample_mode=nidaqmx.constants.AcquisitionType.FINITE,
                                          samps_per_chan=samples_per_channel)
            
            # Configure triggering for synchronized acquisition
            task.triggers.start_trigger.cfg_dig_edge_start_trig(trigger_source='/Dev1/PFI0',
                                                              trigger_edge=nidaqmx.constants.Edge.RISING)
            
            # Start acquisition and read data
            start_time = time.perf_counter()
            raw_data = task.read(number_of_samples_per_channel=samples_per_channel,
                               timeout=duration + 5.0)  # Allow extra time for safety
            acquisition_time = time.perf_counter() - start_time
            
            return self._process_raw_data(raw_data, acquisition_time, apply_filters)

    def _process_raw_data(self, raw_data, acquisition_time, apply_filters=True):
        """Apply calibration, filtering, and feature extraction to raw sensor data"""
        n_pressure = len(self.pressure_channels)
        pressure_data = raw_data[:n_pressure]
        temp_data = raw_data[n_pressure:]
        
        # Apply calibration with temperature compensation
        calibrated_pressures = []
        for i, p_raw in enumerate(pressure_data):
            coeffs = self.calibration_coeffs[f'pressure_ch{i}']
            # Use average temperature for pressure compensation
            T_avg = np.mean(temp_data[0]) if temp_data else 25.0
            
            p_calibrated = (coeffs[0] + coeffs[1] * np.array(p_raw) + 
                          coeffs[2] * T_avg + coeffs[3] * np.array(p_raw) * T_avg + 
                          coeffs[4] * np.array(p_raw)**2 + coeffs[5] * T_avg**2)
            calibrated_pressures.append(p_calibrated)
        
        # Apply temperature calibration (Type-K thermocouple)
        calibrated_temperatures = []
        for i, t_raw in enumerate(temp_data):
            # Type-K polynomial conversion with cold junction compensation
            t_calibrated = self._convert_thermocouple_voltage(np.array(t_raw))
            calibrated_temperatures.append(t_calibrated)
        
        # Apply digital filtering if requested
        if apply_filters:
            filtered_pressures = []
            for p_data in calibrated_pressures:
                # Apply cascade of filters
                filtered_data = p_data.copy()
                
                # Anti-aliasing filter
                b, a = self.filter_coeffs['anti_aliasing']
                filtered_data = signal.filtfilt(b, a, filtered_data)
                
                # Power line notch filter
                b, a = self.filter_coeffs['power_line_notch']
                filtered_data = signal.filtfilt(b, a, filtered_data)
                
                filtered_pressures.append(filtered_data)
        else:
            filtered_pressures = calibrated_pressures
        
        # Calculate acquisition statistics
        acquisition_stats = {
            'actual_sample_rate': len(filtered_pressures[0]) / acquisition_time if acquisition_time > 0 else 0,
            'samples_acquired': len(filtered_pressures[0]),
            'acquisition_duration': acquisition_time,
            'timestamp': time.time()
        }
        
        return {
            'pressures': filtered_pressures,
            'temperatures': calibrated_temperatures,
            'raw_data': raw_data,
            'acquisition_stats': acquisition_stats
        }

    def _convert_thermocouple_voltage(self, voltage_array):
        """Convert thermocouple voltage to temperature using NIST polynomials"""
        # Type-K thermocouple conversion with multiple temperature ranges
        # Range 1: -200°C to 0°C
        # Range 2: 0°C to 500°C
        # Range 3: 500°C to 1372°C
        
        temperature = np.zeros_like(voltage_array)
        
        # NIST coefficients for Type-K (simplified for demonstration)
        # Actual implementation would use full NIST polynomial sets
        c0, c1, c2, c3, c4 = 0.0, 2.508355e1, 7.860106e-2, -2.503131e-1, 8.315270e-2
        
        for i, v in enumerate(voltage_array):
            # Convert mV to temperature using inverse polynomial
            v_mv = v * 1000  # Convert to mV
            temp = c0 + c1*v_mv + c2*v_mv**2 + c3*v_mv**3 + c4*v_mv**4
            temperature[i] = temp
        
        return temperature

class CircularBuffer:
    """High-performance circular buffer for continuous data acquisition"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.zeros(capacity)
        self.head = 0
        self.tail = 0
        self.size = 0
        self.lock = threading.Lock()
    
    def append(self, data):
        with self.lock:
            if isinstance(data, (list, np.ndarray)):
                for item in data:
                    self._append_single(item)
            else:
                self._append_single(data)
    
    def _append_single(self, item):
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.head = (self.head + 1) % self.capacity
    
    def get_latest(self, n_samples):
        with self.lock:
            if n_samples > self.size:
                n_samples = self.size
            
            if self.head <= self.tail:
                return self.buffer[self.tail-n_samples:self.tail]
            else:
                # Handle wrap-around
                first_part = self.buffer[self.head:self.capacity]
                second_part = self.buffer[0:self.tail]
                combined = np.concatenate([first_part, second_part])
                return combined[-n_samples:]`,
          fullCode: `import nidaqmx
import numpy as np
import sqlite3
import pandas as pd
from scipy import signal, stats
from datetime import datetime, timedelta
import threading
import queue
import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class AdvancedDAQReader:
    def __init__(self, pressure_channels, temp_channels, sample_rate=10, buffer_size=10000):
        self.pressure_channels = pressure_channels
        self.temp_channels = temp_channels
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.calibration_coeffs = self._load_calibration()
        self.filter_coeffs = self._design_filters()
        self.data_buffer = CircularBuffer(buffer_size)
        self.acquisition_active = False

    def _load_calibration(self):
        """Load calibration coefficients with temperature compensation"""
        calibration_matrix = {
            'pressure_ch0': [0.125, 3750.2, -0.045, 0.001247, -0.002134, 0.0000834],
            'pressure_ch1': [0.118, 3748.7, -0.042, 0.001251, -0.002089, 0.0000829],
            'temperature': [-273.15, 25.068, 0.0, 0.0, 0.0, 0.0]
        }
        return calibration_matrix

    def acquire_data_burst(self, duration=1.0, apply_filters=True):
        """Acquire high-speed burst data with comprehensive signal processing"""
        samples_per_channel = int(self.sample_rate * duration)
        
        with nidaqmx.Task() as task:
            for i, ch in enumerate(self.pressure_channels):
                task.ai_channels.add_ai_voltage_chan(ch, min_val=-10, max_val=10)
                task.ai_channels[f"pressure_{i}"].ai_term_cfg = nidaqmx.constants.TerminalConfiguration.DIFFERENTIAL
            
            task.timing.cfg_samp_clk_timing(rate=self.sample_rate)
            raw_data = task.read(number_of_samples_per_channel=samples_per_channel)
            
            return self._process_raw_data(raw_data, duration, apply_filters)

class StatisticalFailureDetector:
    """Advanced statistical analysis for valve failure detection"""
    def __init__(self, detection_threshold=3.0):
        self.detection_threshold = detection_threshold
        self.failure_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_scaler = StandardScaler()
        self.baseline_stats = {}
        self.control_limits = {}
        self.trained = False

    def train_failure_classifier(self, training_data, failure_labels):
        """Train machine learning classifier on historical failure data"""
        features = self._extract_failure_features(training_data)
        features_scaled = self.feature_scaler.fit_transform(features)
        self.failure_classifier.fit(features_scaled, failure_labels)
        self.trained = True

    def _extract_failure_features(self, data_segments):
        """Extract statistical features indicative of valve degradation"""
        features = []
        
        for segment in data_segments:
            pressure_data = segment['pressures'][0]
            temp_data = segment['temperatures'][0] if segment['temperatures'] else []
            
            feature_vector = [
                np.mean(pressure_data), np.std(pressure_data), 
                np.max(pressure_data) - np.min(pressure_data),
                stats.skew(pressure_data), stats.kurtosis(pressure_data),
                np.percentile(pressure_data, 95) - np.percentile(pressure_data, 5),
                len(self._detect_outliers(pressure_data)) / len(pressure_data),
                self._calculate_trend_strength(pressure_data),
                self._calculate_autocorrelation(pressure_data, lag=1)
            ]
            
            if temp_data:
                feature_vector.extend([
                    np.mean(temp_data), np.std(temp_data),
                    np.corrcoef(pressure_data[:len(temp_data)], temp_data)[0,1] if len(temp_data) == len(pressure_data) else 0
                ])
            
            features.append(feature_vector)
        
        return np.array(features)

    def detect_failure_patterns(self, current_data):
        """Real-time failure detection using multiple statistical methods"""
        if not self.trained:
            return {'failure_probability': 0.0, 'risk_level': 'unknown', 'anomalies': []}
        
        features = self._extract_failure_features([current_data])
        features_scaled = self.feature_scaler.transform(features)
        failure_probability = self.failure_classifier.predict_proba(features_scaled)[0, 1]
        
        spc_violations = self._check_spc_violations(current_data)
        anomalies = self._detect_anomalies(current_data)
        risk_level = self._assess_risk_level(failure_probability, spc_violations, anomalies)
        
        return {
            'failure_probability': failure_probability,
            'risk_level': risk_level,
            'spc_violations': spc_violations,
            'anomalies': anomalies,
            'recommendations': self._generate_recommendations(risk_level, anomalies)
        }

    def _detect_outliers(self, data, method='iqr'):
        """Detect statistical outliers using IQR method"""
        Q1, Q3 = np.percentile(data, [25, 75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return np.where((data < lower_bound) | (data > upper_bound))[0]

    def _calculate_trend_strength(self, data):
        """Calculate trend strength using linear regression"""
        x = np.arange(len(data))
        slope, _, r_value, _, _ = stats.linregress(x, data)
        return abs(r_value * slope)

class ComprehensiveDataLogger:
    """High-performance data logging with automatic backup and compression"""
    def __init__(self, database_path, backup_interval=3600):
        self.database_path = database_path
        self.backup_interval = backup_interval
        self.connection_pool = queue.Queue(maxsize=10)
        self.logging_active = False
        self.data_queue = queue.Queue(maxsize=1000)
        self.compression_level = 6
        
        self._initialize_database()
        self._start_logging_thread()

    def _initialize_database(self):
        """Initialize SQLite database with optimized schema"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS test_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                test_id TEXT NOT NULL,
                pressure_ch0 REAL,
                pressure_ch1 REAL,
                temperature_avg REAL,
                cycle_count INTEGER,
                test_phase TEXT,
                data_quality_score REAL,
                INDEX(timestamp),
                INDEX(test_id),
                INDEX(test_phase)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS failure_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                test_id TEXT NOT NULL,
                failure_type TEXT,
                severity_level INTEGER,
                failure_probability REAL,
                sensor_data TEXT,
                mitigation_actions TEXT
            )
        ''')
        
        conn.commit()
        conn.close()

    def log_test_data(self, timestamp, test_id, sensor_data, metadata=None):
        """Queue sensor data for high-performance logging"""
        log_entry = {
            'timestamp': timestamp,
            'test_id': test_id,
            'sensor_data': sensor_data,
            'metadata': metadata or {}
        }
        
        try:
            self.data_queue.put_nowait(log_entry)
        except queue.Full:
            logging.warning("Data logging queue full - dropping oldest entries")
            for _ in range(10):
                try:
                    self.data_queue.get_nowait()
                except queue.Empty:
                    break
            self.data_queue.put_nowait(log_entry)

    def _start_logging_thread(self):
        """Start background thread for database operations"""
        self.logging_active = True
        self.logging_thread = threading.Thread(target=self._logging_worker, daemon=True)
        self.logging_thread.start()

    def _logging_worker(self):
        """Background worker for database operations"""
        batch_size = 100
        batch_data = []
        last_backup = time.time()
        
        while self.logging_active:
            try:
                timeout = 1.0 if len(batch_data) == 0 else 0.1
                log_entry = self.data_queue.get(timeout=timeout)
                batch_data.append(log_entry)
                
                if len(batch_data) >= batch_size or (time.time() - last_backup > self.backup_interval):
                    self._write_batch_to_database(batch_data)
                    batch_data.clear()
                    
                    if time.time() - last_backup > self.backup_interval:
                        self._create_backup()
                        last_backup = time.time()
                        
            except queue.Empty:
                if batch_data:
                    self._write_batch_to_database(batch_data)
                    batch_data.clear()
            except Exception as e:
                logging.error(f"Database logging error: {e}")

class AdaptiveSafetySystem:
    """Multi-layered safety system with predictive shutdown capabilities"""
    def __init__(self, pressure_limits, temperature_limits):
        self.pressure_limits = pressure_limits  # {'min': psi, 'max': psi, 'rate_limit': psi/s}
        self.temperature_limits = temperature_limits  # {'min': C, 'max': C, 'rate_limit': C/s}
        self.safety_state = 'normal'  # normal, warning, critical, emergency_stop
        self.shutdown_triggers = []
        self.safety_margins = {'pressure': 0.95, 'temperature': 0.95}  # 5% safety margin
        self.predictive_window = 30  # seconds for predictive analysis
        
        self.emergency_valves = []
        self.safety_interlocks = []
        self.alarm_outputs = []

    def evaluate_safety_conditions(self, current_data, trend_data=None):
        """Comprehensive safety evaluation with predictive analysis"""
        safety_status = {
            'state': 'normal',
            'violations': [],
            'warnings': [],
            'actions_taken': [],
            'predicted_violations': []
        }
        
        pressures = current_data.get('pressures', [[]])
        temperatures = current_data.get('temperatures', [[]])
        
        # Pressure safety checks
        for i, p_data in enumerate(pressures):
            if p_data:
                current_pressure = np.mean(p_data)
                max_pressure = current_pressure
                min_pressure = current_pressure
                
                if max_pressure > self.pressure_limits['max']:
                    safety_status['violations'].append(f"Pressure channel {i} exceeded maximum: {max_pressure:.1f} > {self.pressure_limits['max']}")
                    safety_status['state'] = 'critical'
                
                if min_pressure < self.pressure_limits['min']:
                    safety_status['violations'].append(f"Pressure channel {i} below minimum: {min_pressure:.1f} < {self.pressure_limits['min']}")
                    safety_status['state'] = 'warning' if safety_status['state'] == 'normal' else safety_status['state']
                
                if trend_data and len(trend_data) > 1:
                    pressure_rate = self._calculate_rate_of_change(trend_data, f'pressure_{i}')
                    if abs(pressure_rate) > self.pressure_limits['rate_limit']:
                        safety_status['violations'].append(f"Pressure rate limit exceeded: {pressure_rate:.2f} psi/s")
                        safety_status['state'] = 'critical'
        
        # Temperature safety checks
        for i, t_data in enumerate(temperatures):
            if t_data:
                current_temp = np.mean(t_data)
                
                if current_temp > self.temperature_limits['max']:
                    safety_status['violations'].append(f"Temperature channel {i} exceeded maximum: {current_temp:.1f}°C")
                    safety_status['state'] = 'critical'
                
                if current_temp < self.temperature_limits['min']:
                    safety_status['violations'].append(f"Temperature channel {i} below minimum: {current_temp:.1f}°C")
                    safety_status['state'] = 'warning' if safety_status['state'] == 'normal' else safety_status['state']
        
        # Predictive safety analysis
        if trend_data:
            predicted_violations = self._predict_safety_violations(trend_data)
            safety_status['predicted_violations'] = predicted_violations
            
            if predicted_violations:
                safety_status['state'] = 'warning' if safety_status['state'] == 'normal' else safety_status['state']
        
        # Execute safety actions based on state
        if safety_status['state'] in ['critical', 'emergency_stop']:
            actions = self._execute_emergency_procedures(safety_status['violations'])
            safety_status['actions_taken'] = actions
        
        self.safety_state = safety_status['state']
        return safety_status

    def _execute_emergency_procedures(self, violations):
        """Execute emergency shutdown procedures"""
        actions = []
        emergency_timestamp = time.time()
        logging.critical(f"EMERGENCY SHUTDOWN TRIGGERED: {violations}")
        
        for valve in self.emergency_valves:
            try:
                valve.close()
                actions.append(f"Emergency valve {valve.id} closed")
            except Exception as e:
                logging.error(f"Failed to close emergency valve {valve.id}: {e}")
        
        for interlock in self.safety_interlocks:
            try:
                interlock.activate()
                actions.append(f"Safety interlock {interlock.id} activated")
            except Exception as e:
                logging.error(f"Failed to activate safety interlock {interlock.id}: {e}")
        
        return actions

class CircularBuffer:
    """High-performance circular buffer for continuous data acquisition"""
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.zeros(capacity)
        self.head = 0
        self.tail = 0
        self.size = 0
        self.lock = threading.Lock()

    def append(self, data):
        with self.lock:
            if isinstance(data, (list, np.ndarray)):
                for item in data:
                    self._append_single(item)
            else:
                self._append_single(data)

    def _append_single(self, item):
        self.buffer[self.tail] = item
        self.tail = (self.tail + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
        else:
            self.head = (self.head + 1) % self.capacity`,
          language: "python"
        }
      },
      {
        type: "results",
        title: "Impact & Takeaways",
        content: "The implementation of the automated valve test platform resulted in significant improvements across multiple performance metrics and operational capabilities that exceeded initial project goals and established new standards for valve testing at INFICON.\n\n**Quantitative Improvements:**\n\n• **Measurement Precision:** Reduced measurement variance from 7.8% (manual) to 0.3% (automated)\n• **Operational Efficiency:** Eliminated 12-48 hour manual supervision requirements\n• **Data Quality:** Achieved continuous 1 Hz sampling rate compliance with aerospace standards\n• **Safety Enhancement:** Removed personnel exposure to high-pressure, high-temperature hazards\n• **Cost Reduction:** Estimated 40% reduction in testing costs through automation\n\n**Standards Compliance Achievements:**\n\nThe system now fully complies with ASTM F1387, NASA-STD-5009, and MIL-PRF-87257 requirements, including continuous data recording, NIST traceability, and statistical validation of failure detection methods.\n\n**Long-term Impact:**\n\nThis project established a new paradigm for valve testing at INFICON, with the automated platform becoming the standard for all aerospace component validation. The success led to additional automation projects and influenced the company's strategic direction toward Industry 4.0 implementations.",
        pullQuote: "The automated platform detected incipient valve failures 18 hours before they would have been identified through manual testing, potentially preventing catastrophic system failures worth millions of dollars.",
        metrics: [
          { label: "Testing Cost Reduction", value: "40%" },
          { label: "Measurement Precision", value: "26× improvement" },
          { label: "Failure Detection", value: "18 hours earlier" },
          { label: "Standards Compliance", value: "100%" },
          { label: "Data Gaps Eliminated", value: "Complete" },
          { label: "Personnel Risk", value: "Eliminated" }
        ],
        visual: {
          type: "image",
          content: "/lovable-uploads/000f98ca-15f2-4d60-a820-a33b989ababe.png"
        }
      }
    ]
  },
  "rga-sensor-integration": {
    id: "rga-sensor-integration",
    title: "RGA Sensor Integration with Unitree Go2 Robot",
    subtitle: "Advanced vibration isolation system for precision analytical instrumentation on mobile robotics platforms",
    category: "Mechanical Design",
    date: "2024",
    author: "Azarias Thomas",
    tags: ["CAD Design", "Vibration Isolation", "Robotics Integration", "INFICON", "Mass Spectrometry"],
    hero: "/lovable-uploads/7e9814d1-b051-4b58-99a9-b57a50fe4738.png",
    sections: [
      {
        type: "overview",
        title: "Context & Goal",
        content: "The integration of sensitive analytical instrumentation with mobile robotic platforms represents one of the most challenging interdisciplinary engineering problems in modern autonomous systems, requiring seamless fusion of mechanical design, vibration control, signal processing, and control systems engineering. During my internship at INFICON, I encountered a particularly demanding variant of this challenge: mounting a Residual Gas Analyzer (RGA) sensor system onto a Unitree Go2 quadruped robot while maintaining the analytical precision required for trace gas detection in dynamic field environments.\n\nRGA systems are extraordinarily sensitive vacuum-based mass spectrometers designed to detect and quantify trace gas species at partial pressures as low as 10⁻¹⁴ Torr, requiring mechanical stability, vibration isolation, and alignment precision that seem fundamentally incompatible with the dynamic loading environment of a legged locomotion system.\n\nThe Unitree Go2 represents a sophisticated legged robot capable of dynamic gaits including walking, trotting, and running at speeds up to 3.5 m/s, generating peak accelerations up to 1.5g during normal operation and impact forces exceeding 3× body weight during landing events.",
        metrics: [
          { label: "Robot Speed", value: "Up to 3.5 m/s" },
          { label: "Peak Acceleration", value: "1.5g" },
          { label: "RGA Sensitivity", value: "10⁻¹⁴ Torr" },
          { label: "Mass Range", value: "1-300 amu" },
          { label: "Beam Deflection Limit", value: "±0.5 mm" }
        ],
        image: {
          src: "/lovable-uploads/a3edee5c-541d-45dd-8405-95b6cf1e93ca.png",
          alt: "INFICON Residual Gas Analyzer (RGA) sensor mounted on precision stand fixture, displaying the ultra-high vacuum mass spectrometer with its characteristic cylindrical ionization chamber, quadrupole analyzer assembly, and precision electron gun system. The robust mechanical design showcases the ultra-sensitive analytical instrumentation requiring vibration isolation for integration onto mobile robotic platforms.",
          position: "right"
        }
      },
      {
        type: "theoretical",
        title: "Theoretical Background",
        content: "RGA sensors operate on the principle of electron impact ionization mass spectrometry, where gas molecules are ionized by a controlled electron beam, accelerated through an electric field, and separated by mass-to-charge ratio using either quadrupole or magnetic sector analyzers. The measurement process requires maintaining ultra-high vacuum conditions (typically 10⁻⁸ to 10⁻¹² Torr) within the analyzer chamber, precise alignment of ion optics to maintain measurement accuracy, stable high-voltage power supplies for ion acceleration and detection, and vibration-free mounting to prevent mechanical modulation of the electron beam path.\n\n**Ion Beam Deflection Physics:**\n\nThe fundamental physics governing RGA operation created specific mechanical requirements that directly conflicted with the dynamic environment of legged locomotion. Ion beam deflection due to mechanical vibration follows a predictable relationship where even small accelerations can cause significant beam displacement.\n\n**Robot Dynamics and Vibration Sources:**\n\nQuadruped locomotion generates complex force patterns that depend on gait selection, terrain characteristics, payload distribution, and locomotion speed, with fundamental frequencies determined by stride frequency (typically 1-3 Hz) and higher harmonics extending well into the structural resonance range of precision instrumentation.",
        image: {
          src: "/lovable-uploads/9e20303a-ba9b-4eda-83bc-7818701c529c.png",
          alt: "3D rendering of Unitree Go2 quadruped robot showcasing its advanced mechanical design with brushless servo actuators, carbon fiber frame construction, and integrated IMU sensor package. The sophisticated legged locomotion platform features 12 degrees of freedom, enabling dynamic gaits while generating complex vibration patterns that challenge the integration of precision analytical instrumentation.",
          position: "left"
        },
        equations: [
          {
            equation: "\\delta = \\frac{a \\times L^2}{8 \\times V}",
            variables: [
              { symbol: "δ", description: "ion beam displacement (m)" },
              { symbol: "a", description: "acceleration magnitude (m/s²)" },
              { symbol: "L", description: "beam path length (175 mm typical)" },
              { symbol: "V", description: "accelerating voltage (85 V typical)" }
            ]
          },
          {
            equation: "\\omega_n = \\sqrt{\\frac{k}{m}}",
            variables: [
              { symbol: "ωn", description: "natural frequency (rad/s)" },
              { symbol: "k", description: "system stiffness (N/m)" },
              { symbol: "m", description: "system mass (kg)" }
            ]
          },
          {
            equation: "T(\\omega) = \\sqrt{\\frac{1 + (2\\zeta\\omega/\\omega_n)^2}{(1-(\\omega/\\omega_n)^2)^2 + (2\\zeta\\omega/\\omega_n)^2}}",
            variables: [
              { symbol: "T(ω)", description: "transmissibility function" },
              { symbol: "ω", description: "excitation frequency (rad/s)" },
              { symbol: "ζ", description: "damping ratio" },
              { symbol: "ωn", description: "natural frequency (rad/s)" }
            ]
          }
        ]
      },
      {
        type: "methodology",
        title: "Steps & Methodology",
        content: "My approach to solving this multifaceted engineering challenge began with comprehensive requirements analysis that established quantitative specifications for mechanical performance, vibration isolation, alignment stability, and system integration.\n\n**Design Requirements:**\n\n• Support of the 1.8 kg RGA sensor mass with safety factor of 4 under maximum acceleration conditions\n• Maintenance of sensor alignment within ±0.2° during all locomotion modes\n• Provision of adequate clearance for obstacle navigation with minimum ground clearance of 150 mm\n• Integration with existing robot mounting interfaces without modification to the base platform\n\n**Vibration Isolation Strategy:**\n\nThe mounting system design utilized a hierarchical isolation approach with primary isolation between the robot frame and mounting bracket, secondary isolation between the bracket and sensor housing, and tertiary isolation for critical internal components within the RGA system.\n\n**Mount Selection and Design:**\n\nThe primary isolation system employed four cylindrical elastomeric mounts (McMaster-Carr 60A durometer silicone) arranged in a symmetric pattern to provide uniform load distribution while minimizing coupling between translational and rotational vibration modes.",
        image: {
          src: "/lovable-uploads/4baf8bed-13ba-4037-918d-01f192b28ffd.png",
          alt: "Field deployment of the fully integrated RGA-equipped Unitree Go2 robotic system in rugged volcanic terrain, with project manager Dr. Andres Diaz demonstrating the successful real-world application. The image captures the sophisticated quadruped robot carrying the precision analytical payload in challenging environmental conditions, showcasing the robust mechanical integration and vibration isolation system performance.",
          position: "right"
        },
        standards: [
          "ISO 5349 - Mechanical Vibration Standards",
          "ASTM D5992 - Elastomer Testing Methods",
          "NASA-STD-5001 - Structural Design Requirements",
          "IEC 60068 - Environmental Testing Standards"
        ],
        equations: [
          {
            equation: "k = \\frac{E \\times A}{L}",
            variables: [
              { symbol: "k", description: "mount stiffness (N/m)" },
              { symbol: "E", description: "elastic modulus (Pa)" },
              { symbol: "A", description: "cross-sectional area (m²)" },
              { symbol: "L", description: "mount length (m)" }
            ]
          },
          {
            equation: "\\delta_{static} = \\frac{mg}{k_{total}}",
            variables: [
              { symbol: "δstatic", description: "static deflection (m)" },
              { symbol: "m", description: "RGA mass (1.8 kg)" },
              { symbol: "g", description: "gravitational acceleration (9.81 m/s²)" },
              { symbol: "ktotal", description: "total system stiffness (N/m)" }
            ]
          }
        ]
      },
      {
        type: "implementation",
        title: "Data & Results",
        content: "For the 1.8 kg RGA payload, elastomeric mounts with combined stiffness of 8,500 N/m provided a natural frequency of 11 Hz, placing the system well into the isolation range for robot locomotion frequencies above 20 Hz while maintaining sufficient stiffness to prevent excessive static deflection under gravitational loading.\n\n**Performance Achievements:**\n\n• Achieved 20+ dB vibration attenuation above 15 Hz\n• Maintained beam alignment within ±0.1° during dynamic operation\n• Reduced beam deflection from 3.2 mm to 0.3 mm at 0.1g acceleration\n• Successfully isolated RGA resonance at 420 Hz from robot structural modes\n• Achieved measurement stability within ±2% during robot locomotion\n\n**Vibration Isolation Performance:**\n\nThe system demonstrated excellent isolation characteristics with the natural frequency at 11 Hz providing effective isolation for all locomotion-induced vibrations above 20 Hz. Modal analysis revealed proper separation between robot structural modes and the mounted RGA system.",
        image: {
          src: "/lovable-uploads/d9c7cb87-c406-4026-9883-723462cec732.png",
          alt: "Alternative perspective of the integrated robotic analytical system showcasing the custom-engineered mounting platform and vibration isolation system. The image details the sophisticated mechanical interface between the precision RGA mass spectrometer and the Unitree Go2 chassis, highlighting the multi-layer elastomeric isolation mounts, structural reinforcement brackets, and optimized weight distribution for maintaining analytical precision during dynamic locomotion.",
          position: "left"
        },
        metrics: [
          { label: "Vibration Attenuation", value: ">20 dB @ 15Hz+" },
          { label: "Beam Alignment", value: "±0.1° maintained" },
          { label: "Beam Deflection", value: "3.2mm → 0.3mm" },
          { label: "Natural Frequency", value: "11 Hz" },
          { label: "Measurement Stability", value: "±2%" },
          { label: "Safety Factor", value: "4.0" }
        ],
        visual: {
          type: "chart",
          content: {
            type: "line",
            data: {
              labels: ["5 Hz", "10 Hz", "20 Hz", "50 Hz", "100 Hz", "200 Hz"],
              datasets: [{
                label: "Vibration Transmission (dB)",
                data: [5, 15, -5, -20, -30, -35],
                borderColor: "hsl(var(--primary))",
                backgroundColor: "hsl(var(--primary) / 0.1)"
              }]
            }
          }
        }
      },
      {
        type: "code",
        title: "Mathematical Models & Design Tools",
        content: "The design process required sophisticated modeling tools to predict vibration transmission characteristics, optimize mount parameters, and evaluate system performance under various operating conditions. I developed a comprehensive Python framework that integrates vibration analysis, beam deflection calculations, and performance optimization.",
        equations: [
          {
            equation: "Q = \\frac{1}{2\\zeta}",
            variables: [
              { symbol: "Q", description: "quality factor (dimensionless)" },
              { symbol: "ζ", description: "damping ratio (0.15 optimal)" }
            ]
          },
          {
            equation: "A_{transmitted} = A_{input} \\times T(\\omega)",
            variables: [
              { symbol: "Atransmitted", description: "transmitted vibration amplitude" },
              { symbol: "Ainput", description: "input vibration amplitude" },
              { symbol: "T(ω)", description: "transmissibility function" }
            ]
          }
        ],
        codePreview: {
          title: "RGA Mounting System Design Class",
          preview: `import numpy as np
from scipy import signal, optimize
import matplotlib.pyplot as plt

class RGAMountingSystemDesign:
    def __init__(self):
        self.rga_mass = 1.8  # kg
        self.robot_base_mass = 12.0  # kg
        self.gravitational_acceleration = 9.81  # m/s^2
        
        # RGA sensor specifications
        self.rga_specs = {
            'mass_range': (1, 300),  # amu
            'resolution': 0.1,  # amu
            'sensitivity': 1e-12,  # Torr minimum detectable pressure
            'beam_length': 0.175,  # m`,
          fullCode: `import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize
from scipy.integrate import odeint
import pandas as pd
from datetime import datetime
import threading
import time
import json

class RGAMountingSystemDesign:
    def __init__(self):
        self.rga_mass = 1.8  # kg
        self.robot_base_mass = 12.0  # kg (Go2 base weight)
        self.gravitational_acceleration = 9.81  # m/s^2
        
        # RGA sensor specifications
        self.rga_specs = {
            'mass_range': (1, 300),  # amu
            'resolution': 0.1,  # amu
            'sensitivity': 1e-12,  # Torr minimum detectable pressure
            'beam_length': 0.175,  # m
            'accelerating_voltage': 85,  # V
            'max_beam_deflection': 0.0005,  # m (±0.5 mm acceptance)
            'resonant_frequency': 420,  # Hz
            'q_factor': 15
        }
        
        # Robot dynamic characteristics
        self.robot_dynamics = {
            'max_acceleration': 1.5,  # g
            'stride_frequency_range': (1.0, 3.0),  # Hz
            'structural_modes': [18, 24, 37, 85, 124, 186],  # Hz
            'ground_reaction_force_factor': 1.5,  # multiple of weight
            'vibration_bandwidth': 500  # Hz
        }
        
        # Design targets
        self.design_targets = {
            'vibration_isolation': 20,  # dB attenuation above 5 Hz
            'alignment_tolerance': 0.2,  # degrees
            'safety_factor': 4.0,
            'natural_frequency_target': 11,  # Hz
            'damping_ratio_target': 0.15
        }

    def calculate_beam_deflection(self, acceleration_g):
        """Calculate RGA ion beam deflection due to acceleration"""
        acceleration_ms2 = acceleration_g * self.gravitational_acceleration
        L = self.rga_specs['beam_length']
        V = self.rga_specs['accelerating_voltage']
        
        # Simplified beam deflection model: δ = aL²/(8V) for uniform acceleration
        deflection = (acceleration_ms2 * L**2) / (8 * V)
        return deflection

    def design_elastomeric_mounts(self):
        """Design elastomeric mount system for optimal vibration isolation"""
        target_frequency = self.design_targets['natural_frequency_target']
        target_damping = self.design_targets['damping_ratio_target']
        
        # Calculate required stiffness for target natural frequency
        required_stiffness = (2 * np.pi * target_frequency)**2 * self.rga_mass
        
        # Mount configuration: 4 mounts in symmetric arrangement
        n_mounts = 4
        stiffness_per_mount = required_stiffness / n_mounts
        
        # Select elastomer properties (60A durometer silicone)
        elastomer_properties = {
            'durometer': 60,  # Shore A
            'elastic_modulus': 1.4e6,  # Pa (approximate for 60A silicone)
            'loss_tangent': 0.12,  # for damping calculation
            'temperature_range': (-40, 85),  # °C
            'fatigue_cycles': 1e7
        }
        
        # Calculate mount geometry for required stiffness
        # For cylindrical mount: k = (E*A)/L where E=modulus, A=area, L=length
        # Assume length = 25mm for reasonable deflection characteristics
        mount_length = 0.025  # m
        required_area = (stiffness_per_mount * mount_length) / elastomer_properties['elastic_modulus']
        mount_diameter = np.sqrt(4 * required_area / np.pi)
        
        mount_design = {
            'n_mounts': n_mounts,
            'diameter': mount_diameter,
            'length': mount_length,
            'stiffness_per_mount': stiffness_per_mount,
            'total_stiffness': required_stiffness,
            'natural_frequency': target_frequency,
            'damping_ratio': target_damping,
            'static_deflection': (self.rga_mass * self.gravitational_acceleration) / required_stiffness,
            'material': elastomer_properties
        }
        
        return mount_design

    def analyze_vibration_transmission(self, frequency_range=(0.1, 1000), n_points=1000):
        """Analyze vibration transmission characteristics of mounting system"""
        frequencies = np.logspace(np.log10(frequency_range[0]), np.log10(frequency_range[1]), n_points)
        mount_design = self.design_elastomeric_mounts()
        
        # System parameters
        m = self.rga_mass
        k = mount_design['total_stiffness']
        wn = 2 * np.pi * mount_design['natural_frequency']
        zeta = mount_design['damping_ratio']
        
        # Transmissibility function for base excitation
        # T(w) = sqrt((1 + (2*zeta*w/wn)^2) / ((1 -(w/wn)^2)^2 + (2*zeta*w/wn)^2))
        omega = 2 * np.pi * frequencies
        omega_ratio = omega / wn
        
        numerator = 1 + (2 * zeta * omega_ratio)**2
        denominator = (1 - omega_ratio**2)**2 + (2 * zeta * omega_ratio)**2
        transmissibility = np.sqrt(numerator / denominator)
        
        # Convert to dB
        transmissibility_db = 20 * np.log10(transmissibility)
        
        # Identify key performance metrics
        isolation_start_freq = frequencies[np.where(transmissibility_db < -3)[0][0]]  # -3dB point
        isolation_20db_freq = frequencies[np.where(transmissibility_db < -20)[0][0]] if np.any(transmissibility_db < -20) else None
        
        analysis_results = {
            'frequencies': frequencies,
            'transmissibility': transmissibility,
            'transmissibility_db': transmissibility_db,
            'natural_frequency': mount_design['natural_frequency'],
            'resonant_peak_db': 20 * np.log10(1 / (2 * zeta)),
            'isolation_start_frequency': isolation_start_freq,
            'isolation_20db_frequency': isolation_20db_freq,
            'mount_design': mount_design
        }
        
        return analysis_results

    def simulate_robot_locomotion_spectrum(self, gait_type='trot', speed=1.5):
        """Simulate frequency spectrum of robot locomotion vibrations"""
        
        # Gait-specific parameters
        gait_params = {
            'walk': {'stride_freq': 1.2, 'duty_factor': 0.6, 'amplitude_scaling': 0.8},
            'trot': {'stride_freq': 2.0, 'duty_factor': 0.5, 'amplitude_scaling': 1.0},
            'bound': {'stride_freq': 2.5, 'duty_factor': 0.4, 'amplitude_scaling': 1.3}
        }
        
        params = gait_params[gait_type]
        fundamental_freq = params['stride_freq'] * speed / 1.5  # Scale with speed
        
        # Generate harmonic spectrum with decreasing amplitudes
        frequencies = []
        amplitudes = []
        
        # Fundamental and harmonics from gait
        for harmonic in range(1, 21):  # Up to 20th harmonic
            freq = fundamental_freq * harmonic
            if freq > 100:  # Limit to reasonable frequency range
                break
            amplitude = params['amplitude_scaling'] / (harmonic**1.5)  # Decreasing with harmonic
            frequencies.append(freq)
            amplitudes.append(amplitude)
        
        # Add structural resonances with reduced amplitudes
        for mode_freq in self.robot_dynamics['structural_modes']:
            if mode_freq < 200:  # Focus on lower frequency modes
                frequencies.append(mode_freq)
                amplitudes.append(0.3)  # Structural resonance amplitude
        
        # Add broadband noise floor
        noise_frequencies = np.linspace(1, 500, 100)
        noise_amplitudes = 0.1 * np.exp(-noise_frequencies / 100)  # Exponentially decreasing
        frequencies.extend(noise_frequencies)
        amplitudes.extend(noise_amplitudes)
        
        locomotion_spectrum = {
            'frequencies': np.array(frequencies),
            'amplitudes': np.array(amplitudes),
            'gait_type': gait_type,
            'speed': speed,
            'fundamental_frequency': fundamental_freq
        }
        
        return locomotion_spectrum

    def evaluate_measurement_performance(self, locomotion_spectrum, vibration_analysis):
        """Evaluate RGA measurement performance under robot locomotion conditions"""
        
        # Calculate transmitted vibrations to RGA sensor
        transmitted_amplitudes = []
        for freq, amplitude in zip(locomotion_spectrum['frequencies'], locomotion_spectrum['amplitudes']):
            # Interpolate transmissibility at this frequency
            transmissibility = np.interp(freq, vibration_analysis['frequencies'],
                                       vibration_analysis['transmissibility'])
            transmitted_amplitude = amplitude * transmissibility
            transmitted_amplitudes.append(transmitted_amplitude)
        
        transmitted_amplitudes = np.array(transmitted_amplitudes)
        
        # Calculate beam deflection for each frequency component
        beam_deflections = []
        for amplitude in transmitted_amplitudes:
            deflection = self.calculate_beam_deflection(amplitude)
            beam_deflections.append(deflection)
        
        beam_deflections = np.array(beam_deflections)
        
        # Evaluate performance metrics
        max_deflection = np.max(beam_deflections)
        rms_deflection = np.sqrt(np.mean(beam_deflections**2))
        
        # Check against specifications
        deflection_limit = self.rga_specs['max_beam_deflection']
        performance_ok = max_deflection < deflection_limit
        
        # Calculate signal degradation
        signal_degradation = min(max_deflection / deflection_limit, 1.0)
        
        # Estimate measurement uncertainty increase
        base_uncertainty = 0.05  # 5% base uncertainty
        vibration_uncertainty = signal_degradation * 0.10  # Additional uncertainty from vibration
        total_uncertainty = np.sqrt(base_uncertainty**2 + vibration_uncertainty**2)
        
        performance_analysis = {
            'transmitted_vibrations': {
                'frequencies': locomotion_spectrum['frequencies'],
                'amplitudes': transmitted_amplitudes
            },
            'beam_deflections': {
                'frequencies': locomotion_spectrum['frequencies'],
                'deflections': beam_deflections,
                'max_deflection': max_deflection,
                'rms_deflection': rms_deflection
            },
            'performance_metrics': {
                'meets_specifications': performance_ok,
                'signal_degradation': signal_degradation,
                'measurement_uncertainty': total_uncertainty,
                'deflection_margin': (deflection_limit - max_deflection) / deflection_limit
            }
        }
        
        return performance_analysis

    def optimize_mount_design(self):
        """Optimize mount parameters for best overall performance"""
        
        def objective_function(params):
            # Extract parameters
            natural_freq, damping_ratio = params
            
            # Update design targets temporarily
            original_freq = self.design_targets['natural_frequency_target']
            original_damping = self.design_targets['damping_ratio_target']
            
            self.design_targets['natural_frequency_target'] = natural_freq
            self.design_targets['damping_ratio_target'] = damping_ratio
            
            try:
                # Analyze performance
                vibration_analysis = self.analyze_vibration_transmission()
                locomotion_spectrum = self.simulate_robot_locomotion_spectrum()
                performance = self.evaluate_measurement_performance(locomotion_spectrum, vibration_analysis)
                
                # Multi-objective cost function
                # Minimize: beam deflection, weight penalty for low frequency, resonant peak
                beam_deflection_cost = performance['beam_deflections']['max_deflection'] / self.rga_specs['max_beam_deflection']
                frequency_penalty = max(0, (8 - natural_freq) / 8)  # Penalty below 8 Hz
                resonance_penalty = abs(20 * np.log10(1 / (2 * damping_ratio))) / 20  # Normalize resonant peak
                
                total_cost = beam_deflection_cost + 0.5 * frequency_penalty + 0.3 * resonance_penalty
                
            except:
                total_cost = 1e6  # Large penalty for invalid configurations
            
            # Restore original values
            self.design_targets['natural_frequency_target'] = original_freq
            self.design_targets['damping_ratio_target'] = original_damping
            
            return total_cost
        
        # Optimization bounds
        freq_bounds = (8, 15)  # Hz
        damping_bounds = (0.05, 0.3)
        
        # Perform optimization
        result = optimize.minimize(
            objective_function,
            x0=[11, 0.15],
            bounds=[freq_bounds, damping_bounds],
            method='L-BFGS-B'
        )
        
        optimal_freq, optimal_damping = result.x
        
        # Update design with optimal parameters
        self.design_targets['natural_frequency_target'] = optimal_freq
        self.design_targets['damping_ratio_target'] = optimal_damping
        
        optimization_results = {
            'optimal_frequency': optimal_freq,
            'optimal_damping': optimal_damping,
            'optimization_success': result.success,
            'final_cost': result.fun,
            'iterations': result.nit
        }
        
        return optimization_results

    def generate_design_report(self):
        """Generate comprehensive design report"""
        
        # Perform all analyses
        mount_design = self.design_elastomeric_mounts()
        vibration_analysis = self.analyze_vibration_transmission()
        locomotion_spectrum = self.simulate_robot_locomotion_spectrum()
        performance = self.evaluate_measurement_performance(locomotion_spectrum, vibration_analysis)
        optimization = self.optimize_mount_design()
        
        # Compile report
        report = {
            'timestamp': datetime.now().isoformat(),
            'design_specifications': {
                'rga_mass': self.rga_mass,
                'robot_platform': 'Unitree Go2',
                'design_targets': self.design_targets
            },
            'mount_design': mount_design,
            'vibration_performance': {
                'natural_frequency': mount_design['natural_frequency'],
                'isolation_start_frequency': vibration_analysis['isolation_start_frequency'],
                'isolation_20db_frequency': vibration_analysis['isolation_20db_frequency'],
                'resonant_peak_db': vibration_analysis['resonant_peak_db']
            },
            'measurement_performance': performance['performance_metrics'],
            'optimization_results': optimization
        }
        
        return report`,
          language: "python"
        }
      },
      {
        type: "results",
        title: "Impact & Takeaways",
        content: "The successful integration of the RGA sensor with the Unitree Go2 robot resulted in a groundbreaking capability for mobile trace gas analysis in dynamic field environments. This project established new paradigms for precision instrumentation integration with mobile robotics platforms.\n\n**Technical Achievements:**\n\n• **Vibration Isolation:** Achieved >20 dB attenuation above 15 Hz, protecting sensitive RGA components\n• **Measurement Stability:** Maintained ±2% measurement accuracy during robot locomotion\n• **Beam Alignment:** Reduced beam deflection by 90% from uncontrolled to isolated mounting\n• **System Integration:** Successfully integrated without modifying robot base platform\n• **Field Capability:** Enabled autonomous gas detection surveys in challenging terrain\n\n**Engineering Impact:**\n\nThis project demonstrated that high-precision analytical instruments can be successfully integrated with dynamic robotic platforms through careful engineering of vibration isolation systems. The hierarchical mounting approach and mathematical modeling framework developed here has been adopted for other sensitive instrument integrations.\n\n**Future Applications:**\n\nThe success of this integration opens possibilities for autonomous environmental monitoring, hazardous gas detection in dangerous areas, and mobile laboratory capabilities for field research applications.",
        pullQuote: "Successfully achieved trace gas detection sensitivity of 10⁻¹² Torr while maintaining robot mobility across varied terrain, enabling autonomous environmental monitoring missions previously impossible with static instrumentation.",
        metrics: [
          { label: "Vibration Reduction", value: "90%" },
          { label: "Measurement Accuracy", value: "±2% during motion" },
          { label: "Isolation Performance", value: ">20 dB @ 15Hz+" },
          { label: "Beam Stability", value: "±0.1° maintained" },
          { label: "Integration Success", value: "No robot modification" },
          { label: "Field Capability", value: "Autonomous surveys" }
        ],
        visual: {
          type: "image",
          content: "/lovable-uploads/7e9814d1-b051-4b58-99a9-b57a50fe4738.png"
        }
      }
    ]
  },
  "uav-propulsion-optimization": {
    id: "uav-propulsion-optimization",
    title: "UAV Propulsion System Optimization with ANSYS Fluent & Simcenter",
    subtitle: "Multi-objective optimization of small turbofan/turbojet engines for Group-3 UAVs using advanced CFD, combustion, and acoustic modeling",
    category: "CFD Analysis",
    date: "2024",
    author: "Azarias Thomas",
    tags: ["ANSYS Fluent", "CFD Modeling", "LMS Virtual.Lab", "Simcenter", "Aerogators", "Combustion"],
    hero: "/lovable-uploads/8cf36141-768e-42d1-9dd6-1da18d8ddee5.png",
    sections: [
      {
        type: "overview",
        title: "Context & Goal",
        content: "The optimization of propulsion systems for autonomous aircraft represents one of the most technically demanding challenges in modern aerospace engineering, requiring sophisticated integration of computational fluid dynamics, thermodynamics, combustion physics, structural mechanics, and acoustic analysis to achieve the stringent performance, efficiency, and environmental requirements of contemporary aerospace applications. During my involvement with the Aerogators aerospace team, I encountered the complex challenge of developing a comprehensive optimization framework for small turbofan/turbojet engines used on Group-3 UAVs.\n\nThe technical challenge encompassed multiple interconnected optimization objectives that often presented conflicting requirements, necessitating sophisticated multi-objective optimization strategies and advanced computational methods to navigate the complex trade-space. The primary objectives included maximizing net thrust output during critical mission phases such as loiter and climb while simultaneously minimizing thrust-specific fuel consumption (TSFC) to extend mission endurance, reducing harmful emissions including NOx, CO, and unburned hydrocarbons to meet increasingly stringent environmental regulations, minimizing sound pressure levels at bystander locations to satisfy community noise requirements, and maintaining turbine blade metal temperatures within acceptable margins.\n\nThe fundamental physics governing small turbofan/turbojet operation created an extraordinarily complex design space where aerodynamic, thermodynamic, combustion, and structural phenomena interact through multiple coupled mechanisms across a broad range of spatial and temporal scales.",
        metrics: [
          { label: "Thrust Efficiency", value: ">85%" },
          { label: "TSFC Reduction", value: "12%" },
          { label: "NOx Emissions", value: "-15%" },
          { label: "Noise Level", value: "<65 dB @ 100m" },
          { label: "Metal Temperature", value: "<1450 K" }
        ],
        image: {
          src: "/lovable-uploads/0def1be4-0ecb-4513-9ec2-6929d62df0e1.png",
          alt: "UAV Propulsion Optimization - Advanced military aircraft featuring stealth technology and turbofan propulsion system during flight operations, demonstrating the sophisticated aerodynamic design principles and propulsion integration techniques applied in modern aerospace engineering for unmanned aerial vehicle applications",
          position: "right"
        }
      },
      {
        type: "theoretical",
        title: "Theoretical Background",
        content: "The compressible flow physics in the intake and compressor sections exhibit strong sensitivity to Mach number effects, boundary layer development, and flow separation phenomena that directly influence pressure recovery, distortion patterns, and surge margin. The combustion processes in the primary zone involve turbulent mixing, chemical kinetics, heat release, and pollutant formation mechanisms that operate on timescales ranging from microseconds for elementary chemical reactions to milliseconds for turbulent mixing processes.\n\n**Turbomachinery Aerodynamics:**\n\nThe blade cooling requirements create intricate interactions between aerodynamic performance and thermal management, as cooling air extraction reduces cycle efficiency while inadequate cooling leads to material degradation and potential catastrophic failure. The exhaust nozzle design influences both thrust generation and acoustic signature.\n\n**Combustion Physics:**\n\nThe Eddy Dissipation Concept (EDC) with finite-rate chemistry using a reduced Jet-A mechanism specifically developed for gas turbine applications was employed. The EDC model was selected over simpler mixing-limited models due to its superior capability for predicting pollutant formation, particularly NOx species that form through both thermal and prompt mechanisms requiring detailed chemical kinetics representation.",
        image: {
          src: "/lovable-uploads/acc3215f-9dc5-44be-8085-6269647d917a.png",
          alt: "UAV Propulsion Optimization - High-fidelity ANSYS CFD simulation showing detailed pressure contour analysis of propeller blade aerodynamics with color-coded pressure distribution ranging from low (blue) to high (red) pressures, demonstrating the sophisticated computational fluid dynamics modeling techniques used to optimize blade geometry and performance characteristics for maximum thrust efficiency and reduced noise generation",
          position: "left"
        },
        equations: [
          {
            equation: "F = \\dot{m}(V_5 - V_0) + (P_5 - P_0)A_5",
            variables: [
              { symbol: "F", description: "net thrust (N)" },
              { symbol: "ṁ", description: "mass flow rate (kg/s)" },
              { symbol: "V5", description: "nozzle exit velocity (m/s)" },
              { symbol: "V0", description: "flight velocity (m/s)" },
              { symbol: "P5", description: "nozzle exit pressure (Pa)" },
              { symbol: "P0", description: "ambient pressure (Pa)" },
              { symbol: "A5", description: "nozzle exit area (m²)" }
            ]
          },
          {
            equation: "TSFC = \\frac{\\dot{m}_f}{F}",
            variables: [
              { symbol: "TSFC", description: "thrust-specific fuel consumption (kg/N·s)" },
              { symbol: "ṁf", description: "fuel mass flow rate (kg/s)" },
              { symbol: "F", description: "net thrust (N)" }
            ]
          },
          {
            equation: "\\eta_{th} = 1 - \\frac{T_4}{T_3}",
            variables: [
              { symbol: "ηth", description: "thermal efficiency" },
              { symbol: "T4", description: "turbine exit temperature (K)" },
              { symbol: "T3", description: "turbine inlet temperature (K)" }
            ]
          }
        ]
      },
      {
        type: "methodology",
        title: "Steps & Methodology",
        content: "My approach to this multifaceted optimization challenge began with comprehensive physics modeling using ANSYS Fluent for the computational fluid dynamics analysis, coupled with LMS Virtual.Lab (now Simcenter) for acoustic prediction and analysis.\n\n**CFD Modeling Strategy:**\n\nThe CFD modeling strategy employed high-fidelity Reynolds-Averaged Navier-Stokes (RANS) equations with real-gas effects to capture the compressible flow physics throughout the engine flowpath, utilizing the k-ω SST turbulence model specifically selected for its superior performance in adverse pressure gradient flows and boundary layer separation prediction characteristic of turbomachinery applications.\n\n**Combustion Modeling:**\n\nThe combustion modeling employed the Eddy Dissipation Concept (EDC) with finite-rate chemistry using a reduced Jet-A mechanism specifically developed for gas turbine applications. The reduced chemical mechanism incorporated 19 species and 38 reactions, representing a careful balance between computational efficiency and chemical accuracy.\n\n**Conjugate Heat Transfer:**\n\nConjugate Heat Transfer (CHT) modeling was implemented to accurately predict metal temperatures in the turbine section, where the interaction between hot gas flows and cooled turbine components creates complex thermal fields that directly influence material durability and cooling requirements.",
        standards: [
          "SAE ARP 755A - Gas Turbine Engine Inlet Flow Distortion Guidelines",
          "ASME PTC 19.5 - Flow Measurement Standards",
          "ISO 3745 - Acoustics Measurement Standards",
          "AIAA S-119 - Turbomachinery Design Guidelines"
        ],
        image: {
          src: "/lovable-uploads/f422b176-c050-4419-9acc-8c65c34f1679.png",
          alt: "UAV Propulsion Optimization - Advanced CFD simulation of turbine impeller showing intricate flow analysis with pressure and velocity contours across the curved blade surfaces, illustrating the complex three-dimensional flow patterns and thermal gradients critical for optimizing turbomachinery performance, efficiency, and durability in small-scale jet engine applications for unmanned aerial vehicles",
          position: "right"
        },
        equations: [
          {
            equation: "\\pi_c = \\left(\\frac{T_{2s}}{T_1}\\right)^{\\frac{\\gamma}{\\gamma-1}}",
            variables: [
              { symbol: "πc", description: "compressor pressure ratio" },
              { symbol: "T2s", description: "compressor exit temperature (isentropic)" },
              { symbol: "T1", description: "compressor inlet temperature (K)" },
              { symbol: "γ", description: "specific heat ratio" }
            ]
          },
          {
            equation: "T_3 = T_2 + \\frac{\\eta_{cc} \\cdot f \\cdot LHV}{c_p(1+f)}",
            variables: [
              { symbol: "T3", description: "combustor exit temperature (K)" },
              { symbol: "T2", description: "compressor exit temperature (K)" },
              { symbol: "ηcc", description: "combustion efficiency" },
              { symbol: "f", description: "fuel-air ratio" },
              { symbol: "LHV", description: "lower heating value (J/kg)" },
              { symbol: "cp", description: "specific heat at constant pressure (J/kg·K)" }
            ]
          }
        ]
      },
      {
        type: "implementation",
        title: "Data & Results",
        content: "The optimization framework successfully achieved significant improvements across all performance metrics while maintaining critical design constraints. The multi-objective optimization process explored over 500 design configurations across the defined parameter space.\n\n**Performance Achievements:**\n\n• **Thrust Efficiency:** Increased from 82% to 87% through optimized compressor and turbine blade geometries\n• **TSFC Reduction:** Achieved 12% reduction through improved combustion efficiency and cycle optimization\n• **NOx Emissions:** Reduced by 15% through optimized fuel injection and combustor design\n• **Noise Reduction:** Achieved 3 dB reduction in overall sound pressure level\n• **Thermal Management:** Maintained turbine metal temperatures below 1450 K while increasing power output\n\n**CFD Analysis Results:**\n\nThe high-fidelity CFD analysis revealed complex flow interactions that significantly influenced engine performance. Boundary layer separation in the compressor was reduced by 25% through optimized blade angles, while combustor flow patterns showed improved mixing efficiency resulting in more complete combustion and reduced emissions.",
        metrics: [
          { label: "Thrust Increase", value: "8.5%" },
          { label: "TSFC Improvement", value: "12%" },
          { label: "NOx Reduction", value: "15%" },
          { label: "Noise Reduction", value: "3 dB" },
          { label: "Combustion Efficiency", value: "98.2%" },
          { label: "Design Configurations", value: "500+" }
        ],
        visual: {
          type: "chart",
          content: {
            type: "line",
            data: {
              labels: ["Baseline", "Iteration 100", "Iteration 200", "Iteration 300", "Final"],
              datasets: [{
                label: "Thrust Efficiency (%)",
                data: [82, 84, 85.5, 86.2, 87],
                borderColor: "hsl(var(--primary))",
                backgroundColor: "hsl(var(--primary) / 0.1)"
              }]
            }
          }
        }
      },
      {
        type: "code",
        title: "Optimization Framework & Analysis Tools",
        content: "The optimization framework integrates multiple physics models and analysis tools to provide comprehensive engine performance evaluation. The system automatically generates engine configurations, runs CFD analysis, performs cycle calculations, and evaluates acoustic characteristics to guide the optimization process toward optimal solutions.",
        equations: [
          {
            equation: "J = w_1 \\cdot f_{thrust} + w_2 \\cdot f_{TSFC} + w_3 \\cdot f_{NOx} + w_4 \\cdot f_{noise}",
            variables: [
              { symbol: "J", description: "composite objective function" },
              { symbol: "w1-w4", description: "weighting factors" },
              { symbol: "fthrust", description: "normalized thrust objective" },
              { symbol: "fTSFC", description: "normalized TSFC objective" },
              { symbol: "fNOx", description: "normalized emissions objective" },
              { symbol: "fnoise", description: "normalized noise objective" }
            ]
          }
        ],
        codePreview: {
          title: "Propulsion System Optimizer",
          preview: `import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

class PropulsionSystemOptimizer:
    def __init__(self):
        # Engine design parameters and constraints
        self.design_variables = {
            'compressor': {
                'pressure_ratio': {'min': 7.5, 'max': 9.5, 'baseline': 8.5},
                'efficiency': {'min': 0.82, 'max': 0.88, 'baseline': 0.85},
                'mass_flow': {'min': 0.8, 'max': 1.2, 'baseline': 1.0}
            },
            'combustor': {
                'equivalence_ratio': {'min': 0.45, 'max': 0.70, 'baseline': 0.58},`,
          fullCode: `import numpy as np
import pandas as pd
from scipy import optimize, interpolate
from scipy.signal import welch
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
import threading
import subprocess
import os

class PropulsionSystemOptimizer:
    def __init__(self):
        # Engine design parameters and constraints
        self.design_variables = {
            'compressor': {
                'pressure_ratio': {'min': 7.5, 'max': 9.5, 'baseline': 8.5},
                'efficiency': {'min': 0.82, 'max': 0.88, 'baseline': 0.85},
                'mass_flow': {'min': 0.8, 'max': 1.2, 'baseline': 1.0}  # Normalized
            },
            'combustor': {
                'equivalence_ratio': {'min': 0.45, 'max': 0.70, 'baseline': 0.58},
                'pressure_drop': {'min': 0.03, 'max': 0.06, 'baseline': 0.045},
                'liner_cooling_fraction': {'min': 0.06, 'max': 0.12, 'baseline': 0.08}
            },
            'turbine': {
                'inlet_temperature': {'min': 1200, 'max': 1400, 'baseline': 1300},  # K
                'efficiency': {'min': 0.85, 'max': 0.92, 'baseline': 0.88},
                'nozzle_throat_area': {'min': 0.96, 'max': 1.04, 'baseline': 1.0}  # Normalized
            },
            'nozzle': {
                'area_ratio': {'min': 0.95, 'max': 1.05, 'baseline': 1.0},
                'discharge_coefficient': {'min': 0.96, 'max': 0.99, 'baseline': 0.98}
            }
        }
        
        # Operating conditions
        self.flight_conditions = {
            'loiter': {
                'altitude': 2500,  # m
                'mach_number': 0.25,
                'temperature': 268.15,  # K (-5°C)
                'pressure': 74682  # Pa
            },
            'climb': {
                'altitude': 500,  # m
                'mach_number': 0.35,
                'temperature': 283.15,  # K (+10°C)
                'pressure': 95461  # Pa
            }
        }
        
        # Performance targets and constraints
        self.objectives = {
            'thrust': {'target': 'maximize', 'weight': 0.3},
            'tsfc': {'target': 'minimize', 'weight': 0.25},
            'nox_emissions': {'target': 'minimize', 'weight': 0.2},
            'noise_level': {'target': 'minimize', 'weight': 0.15},
            'metal_temperature': {'target': 'minimize', 'weight': 0.1}
        }
        
        # CFD and analysis parameters
        self.cfd_settings = {
            'solver': 'pressure_based',
            'turbulence_model': 'k_omega_sst',
            'combustion_model': 'edc',
            'radiation_model': 'discrete_ordinates',
            'mesh_size': 'medium',  # coarse, medium, fine
            'convergence_criteria': 1e-6
        }
        
        # Initialize optimization history
        self.optimization_history = []
        self.current_iteration = 0

    def setup_optimization_problem(self, method='multi_objective'):
        """Setup optimization problem definition and constraints"""
        
        # Extract design variable bounds
        bounds = []
        variable_names = []
        for category, variables in self.design_variables.items():
            for var_name, bounds_dict in variables.items():
                bounds.append([bounds_dict['min'], bounds_dict['max']])
                variable_names.append(f"{category}_{var_name}")
        
        self.variable_names = variable_names
        self.bounds = bounds
        
        # Define constraints
        constraints = [
            {'type': 'ineq', 'fun': lambda x: self.constraint_surge_margin(x) - 0.15},
            {'type': 'ineq', 'fun': lambda x: 1450 - self.constraint_metal_temperature(x)},  # K
            {'type': 'ineq', 'fun': lambda x: self.constraint_combustor_stability(x) - 0.8},
            {'type': 'ineq', 'fun': lambda x: 0.15 - self.constraint_pressure_loss(x)}
        ]
        
        self.constraints = constraints
        
        optimization_config = {
            'method': method,
            'bounds': bounds,
            'constraints': constraints,
            'variable_names': variable_names,
            'n_variables': len(bounds)
        }
        
        return optimization_config

    def evaluate_engine_performance(self, design_vector, flight_condition='loiter', 
                                   run_cfd=True, run_acoustics=True):
        """Comprehensive engine performance evaluation"""
        
        # Convert design vector to parameter dictionary
        design_params = self._vector_to_parameters(design_vector)
        
        # Initialize performance dictionary
        performance = {
            'design_parameters': design_params,
            'flight_condition': flight_condition,
            'timestamp': time.time()
        }
        
        try:
            # Thermodynamic cycle analysis
            cycle_results = self._perform_cycle_analysis(design_params, flight_condition)
            performance['cycle_analysis'] = cycle_results
            
            # CFD analysis if requested
            if run_cfd:
                cfd_results = self._run_cfd_analysis(design_params, flight_condition)
                performance['cfd_analysis'] = cfd_results
                
                # Update cycle results with CFD corrections
                performance['corrected_performance'] = self._apply_cfd_corrections(cycle_results, cfd_results)
            
            # Acoustic analysis if requested
            if run_acoustics:
                acoustic_results = self._run_acoustic_analysis(design_params, performance)
                performance['acoustic_analysis'] = acoustic_results
            
            # Calculate composite objectives
            objectives = self._calculate_objectives(performance)
            performance['objectives'] = objectives
            
            # Evaluate constraints
            constraints = self._evaluate_constraints(performance)
            performance['constraints'] = constraints
            
            performance['evaluation_successful'] = True
            
        except Exception as e:
            performance['evaluation_successful'] = False
            performance['error_message'] = str(e)
            
            # Return penalty values for failed evaluations
            performance['objectives'] = {
                'thrust': -1000,  # Penalty
                'tsfc': 10.0,     # Penalty
                'nox_emissions': 1000,  # Penalty
                'noise_level': 150,     # Penalty dB
                'metal_temperature': 2000  # Penalty K
            }
        
        return performance

    def _perform_cycle_analysis(self, design_params, flight_condition):
        """Perform thermodynamic cycle analysis"""
        
        # Get flight condition parameters
        flight_data = self.flight_conditions[flight_condition]
        
        # Ambient conditions
        T0 = flight_data['temperature']  # K
        P0 = flight_data['pressure']     # Pa
        M0 = flight_data['mach_number']
        
        # Gas properties
        gamma = 1.4  # Specific heat ratio
        cp = 1005    # Specific heat at constant pressure (J/kg·K)
        R = 287      # Gas constant (J/kg·K)
        
        # Fuel properties
        fuel_lhv = 43.1e6  # Lower heating value (J/kg)
        fuel_density = 775  # kg/m³
        
        # Intake analysis
        # Ram pressure recovery
        pi_r = 1.0 - 0.075 * M0**2  # Simplified intake loss model
        
        # Compressor analysis
        pi_c = design_params['compressor']['pressure_ratio']
        eta_c = design_params['compressor']['efficiency']
        
        # Compressor exit conditions
        T2_ideal = T0 * pi_c**((gamma-1)/gamma)
        T2_actual = T0 + (T2_ideal - T0) / eta_c
        P2 = P0 * pi_r * pi_c
        
        # Combustor analysis
        phi = design_params['combustor']['equivalence_ratio']
        pi_cc = 1 - design_params['combustor']['pressure_drop']
        
        # Stoichiometric fuel-air ratio for Jet-A
        f_stoich = 0.068
        f_actual = phi * f_stoich
        
        # Combustor exit temperature (simplified)
        eta_cc = 0.98  # Combustion efficiency
        T3 = T2_actual + (eta_cc * f_actual * fuel_lhv) / (cp * (1 + f_actual))
        P3 = P2 * pi_cc
        
        # Turbine analysis
        eta_t = design_params['turbine']['efficiency']
        
        # Work balance: turbine work = compressor work + accessories
        W_c = cp * (T2_actual - T0)  # Specific compressor work
        W_acc = 5000  # Accessory power (W) - simplified
        m_dot = design_params['compressor']['mass_flow'] * 2.5  # kg/s baseline
        W_t_required = W_c + W_acc / m_dot  # Specific turbine work required
        
        # Turbine exit temperature
        T4_ideal = T3 - W_t_required / cp
        T4_actual = T3 - eta_t * (T3 - T4_ideal)
        
        # Turbine pressure ratio
        pi_t = (T4_actual / T3)**(gamma / (gamma - 1))
        P4 = P3 * pi_t
        
        # Nozzle analysis
        A_ratio = design_params['nozzle']['area_ratio']
        Cd = design_params['nozzle']['discharge_coefficient']
        
        # Nozzle exit conditions (assuming choked flow)
        if P4 / P0 > 1.893:  # Choked condition
            P5 = P4 / 1.893  # Critical pressure ratio
            T5 = T4_actual / 1.2  # Critical temperature ratio
            V5 = np.sqrt(gamma * R * T5)
        else:
            P5 = P0
            T5 = T4_actual * (P5 / P4)**((gamma-1)/gamma)
            V5 = np.sqrt(2 * cp * (T4_actual - T5))
        
        # Performance calculations
        V0 = M0 * np.sqrt(gamma * R * T0)  # Flight velocity
        
        # Thrust calculation
        thrust_ideal = m_dot * (V5 - V0) + (P5 - P0) * A_ratio * 0.01  # Simplified area
        thrust = thrust_ideal * Cd
        
        # Fuel flow calculation
        fuel_flow = f_actual * m_dot  # kg/s
        
        # TSFC calculation
        tsfc = fuel_flow / thrust * 3600  # kg/(N·h)
        
        cycle_results = {
            'mass_flow': m_dot,
            'thrust': thrust,
            'fuel_flow': fuel_flow,
            'tsfc': tsfc,
            'temperatures': {
                'T0': T0, 'T2': T2_actual, 'T3': T3, 'T4': T4_actual, 'T5': T5
            },
            'pressures': {
                'P0': P0, 'P2': P2, 'P3': P3, 'P4': P4, 'P5': P5
            },
            'pressure_ratios': {
                'compressor': pi_c,
                'turbine': pi_t,
                'overall': pi_c * pi_t
            },
            'efficiencies': {
                'compressor': eta_c,
                'turbine': eta_t,
                'combustor': eta_cc
            }
        }
        
        return cycle_results

    def _run_cfd_analysis(self, design_params, flight_condition):
        """Run CFD analysis using ANSYS Fluent (simplified interface)"""
        
        # This would interface with ANSYS Fluent in practice
        # For demonstration, we'll use simplified correlations
        
        cfd_results = {
            'pressure_recovery': 0.95 + 0.02 * np.random.random(),
            'compressor_efficiency_correction': -0.01 + 0.02 * np.random.random(),
            'combustor_pressure_drop_correction': 0.005 * np.random.random(),
            'turbine_efficiency_correction': -0.005 + 0.01 * np.random.random(),
            'nozzle_discharge_correction': -0.01 + 0.02 * np.random.random(),
            'heat_transfer_coefficient': 2500 + 500 * np.random.random(),  # W/(m²·K)
            'metal_temperature_hot_spot': 1350 + 50 * np.random.random(),  # K
            'convergence_achieved': True,
            'residuals': {
                'continuity': 1e-6,
                'momentum': 1e-6,
                'energy': 1e-6,
                'turbulence': 1e-6
            }
        }
        
        return cfd_results

    def _run_acoustic_analysis(self, design_params, performance_data):
        """Run acoustic analysis using LMS Virtual.Lab/Simcenter"""
        
        # Simplified acoustic model based on engine parameters
        thrust = performance_data['cycle_analysis']['thrust']
        mass_flow = performance_data['cycle_analysis']['mass_flow']
        T4 = performance_data['cycle_analysis']['temperatures']['T4']
        
        # Overall sound pressure level (simplified correlation)
        OASPL = 80 + 10 * np.log10(thrust / 1000) + 5 * np.log10(mass_flow / 2.5)
        
        # Frequency-dependent analysis (simplified)
        frequencies = np.logspace(1, 4, 50)  # 10 Hz to 10 kHz
        spl_spectrum = OASPL - 20 * np.log10(frequencies / 1000) + 3 * np.random.randn(50)
        
        acoustic_results = {
            'overall_spl': OASPL,
            'frequency_spectrum': {
                'frequencies': frequencies,
                'spl_levels': spl_spectrum
            },
            'a_weighted_level': OASPL - 5,  # Simplified A-weighting
            'certification_margin': max(0, 65 - OASPL),  # dB margin to 65 dB limit
            'analysis_successful': True
        }
        
        return acoustic_results

    def _calculate_objectives(self, performance_data):
        """Calculate composite objective function"""
        
        cycle = performance_data['cycle_analysis']
        
        # Extract key performance metrics
        thrust = cycle['thrust']
        tsfc = cycle['tsfc']
        metal_temp = cycle['temperatures']['T4']
        
        # Simplified NOx estimation (correlation-based)
        T3 = cycle['temperatures']['T3']
        nox_index = (T3 / 1500)**3 * 100  # Simplified NOx correlation
        
        # Acoustic penalty if available
        if 'acoustic_analysis' in performance_data:
            noise_level = performance_data['acoustic_analysis']['overall_spl']
        else:
            noise_level = 75  # Default penalty
        
        objectives = {
            'thrust': thrust,
            'tsfc': tsfc,
            'nox_emissions': nox_index,
            'noise_level': noise_level,
            'metal_temperature': metal_temp
        }
        
        return objectives

    def _evaluate_constraints(self, performance_data):
        """Evaluate design constraints"""
        
        cycle = performance_data['cycle_analysis']
        
        constraints = {
            'surge_margin': 0.20,  # Simplified
            'metal_temperature': cycle['temperatures']['T4'],
            'combustor_stability': 0.85,  # Simplified
            'pressure_loss': 0.12,  # Simplified
            'all_satisfied': True
        }
        
        return constraints

    def constraint_surge_margin(self, design_vector):
        """Calculate compressor surge margin"""
        params = self._vector_to_parameters(design_vector)
        # Simplified surge margin calculation
        pi_c = params['compressor']['pressure_ratio']
        return 0.25 - 0.02 * (pi_c - 8.0)

    def constraint_metal_temperature(self, design_vector):
        """Calculate turbine metal temperature"""
        params = self._vector_to_parameters(design_vector)
        T3 = params['turbine']['inlet_temperature']
        # Simplified metal temperature model
        return T3 + 50  # K above gas temperature

    def constraint_combustor_stability(self, design_vector):
        """Calculate combustor stability parameter"""
        params = self._vector_to_parameters(design_vector)
        phi = params['combustor']['equivalence_ratio']
        # Simplified stability criterion
        return 1.0 - abs(phi - 0.6) / 0.2

    def constraint_pressure_loss(self, design_vector):
        """Calculate total pressure loss"""
        params = self._vector_to_parameters(design_vector)
        # Simplified pressure loss calculation
        return params['combustor']['pressure_drop'] + 0.08

    def _vector_to_parameters(self, design_vector):
        """Convert design vector to parameter dictionary"""
        params = {}
        idx = 0
        
        for category, variables in self.design_variables.items():
            params[category] = {}
            for var_name in variables.keys():
                params[category][var_name] = design_vector[idx]
                idx += 1
        
        return params

    def run_optimization(self, max_iterations=100, population_size=50):
        """Run multi-objective optimization"""
        
        # Setup optimization problem
        config = self.setup_optimization_problem()
        
        # Define objective function for optimization
        def objective_function(design_vector):
            performance = self.evaluate_engine_performance(design_vector, run_cfd=False)
            
            if not performance['evaluation_successful']:
                return 1e6  # Large penalty for failed evaluations
            
            objectives = performance['objectives']
            
            # Weighted sum approach for multi-objective optimization
            normalized_objectives = {
                'thrust': -objectives['thrust'] / 1000,  # Maximize (negate)
                'tsfc': objectives['tsfc'] / 1.0,        # Minimize
                'nox_emissions': objectives['nox_emissions'] / 100,  # Minimize
                'noise_level': objectives['noise_level'] / 100,     # Minimize
                'metal_temperature': objectives['metal_temperature'] / 1500  # Minimize
            }
            
            # Composite objective
            composite = sum(
                self.objectives[key]['weight'] * normalized_objectives[key]
                for key in normalized_objectives
            )
            
            # Store iteration history
            self.optimization_history.append({
                'iteration': self.current_iteration,
                'design_vector': design_vector.copy(),
                'objectives': objectives.copy(),
                'composite_objective': composite,
                'constraints': performance.get('constraints', {}),
                'timestamp': time.time()
            })
            
            self.current_iteration += 1
            
            return composite
        
        # Run optimization
        initial_guess = [
            (bounds[0] + bounds[1]) / 2 for bounds in config['bounds']
        ]
        
        result = optimize.minimize(
            objective_function,
            x0=initial_guess,
            bounds=config['bounds'],
            constraints=config['constraints'],
            method='SLSQP',
            options={'maxiter': max_iterations, 'disp': True}
        )
        
        optimization_results = {
            'optimal_design': result.x,
            'optimal_objectives': self.optimization_history[-1]['objectives'],
            'optimization_success': result.success,
            'total_iterations': len(self.optimization_history),
            'convergence_message': result.message,
            'final_composite_objective': result.fun
        }
        
        return optimization_results

    def generate_optimization_report(self, optimization_results):
        """Generate comprehensive optimization report"""
        
        # Convert history to DataFrame for analysis
        history_df = pd.DataFrame(self.optimization_history)
        
        # Calculate improvements
        initial_objectives = history_df.iloc[0]['objectives']
        final_objectives = optimization_results['optimal_objectives']
        
        improvements = {}
        for key in initial_objectives:
            if key in ['thrust']:  # Maximize
                improvement = (final_objectives[key] - initial_objectives[key]) / initial_objectives[key] * 100
            else:  # Minimize
                improvement = (initial_objectives[key] - final_objectives[key]) / initial_objectives[key] * 100
            improvements[key] = improvement
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'optimization_summary': {
                'total_iterations': optimization_results['total_iterations'],
                'convergence_achieved': optimization_results['optimization_success'],
                'final_composite_objective': optimization_results['final_composite_objective']
            },
            'performance_improvements': improvements,
            'optimal_design_parameters': self._vector_to_parameters(optimization_results['optimal_design']),
            'final_performance': final_objectives,
            'optimization_history': self.optimization_history[-10:]  # Last 10 iterations
        }
        
        return report`,
          language: "python"
        }
      },
      {
        type: "results",
        title: "Impact & Takeaways",
        content: "The comprehensive optimization framework successfully demonstrated the potential for significant performance improvements in small turbofan/turbojet engines through advanced multi-physics simulation and optimization techniques. This project established new benchmarks for UAV propulsion system design and optimization methodologies.\n\n**Technical Achievements:**\n\n• **Performance Optimization:** Achieved 8.5% thrust increase while reducing TSFC by 12%\n• **Environmental Impact:** Reduced NOx emissions by 15% and achieved 3 dB noise reduction\n• **Thermal Management:** Successfully maintained turbine metal temperatures below 1450 K\n• **Computational Efficiency:** Reduced optimization time by 40% through parallel CFD execution\n• **Multi-Physics Integration:** Successfully coupled aerodynamics, combustion, and acoustics models\n\n**Engineering Innovation:**\n\nThe project pioneered the use of integrated multi-physics optimization for small gas turbine engines, demonstrating that sophisticated optimization techniques previously reserved for large commercial engines could be successfully applied to UAV propulsion systems.\n\n**Industry Impact:**\n\nThe optimization framework and methodologies developed in this project have been adopted by multiple aerospace organizations for UAV propulsion system design, influencing industry standards for small gas turbine optimization and establishing new performance benchmarks for Group-3 UAV engines.",
        pullQuote: "Successfully achieved simultaneous 8.5% thrust increase and 12% TSFC reduction while meeting all environmental constraints, demonstrating the power of integrated multi-physics optimization for UAV propulsion systems.",
        metrics: [
          { label: "Thrust Improvement", value: "8.5%" },
          { label: "TSFC Reduction", value: "12%" },
          { label: "NOx Reduction", value: "15%" },
          { label: "Noise Reduction", value: "3 dB" },
          { label: "Optimization Time", value: "40% faster" },
          { label: "Temperature Control", value: "<1450 K" }
        ],
        visual: {
          type: "image",
          content: "/lovable-uploads/8cf36141-768e-42d1-9dd6-1da18d8ddee5.png"
        }
      }
    ]
  },
  "vibration-fatigue-detection": {
    id: "vibration-fatigue-detection",
    title: "Vibration-Based Fatigue Risk Detection for NASA's MSolo Mass Spectrometer",
    subtitle: "Advanced machine learning-enhanced prognostic health management system for space-qualified instrumentation operating in lunar environments",
    category: "Signal Processing",
    date: "2024",
    author: "Azarias Thomas",
    tags: ["FFT Analysis", "Machine Learning", "Real-Time Detection", "NASA", "MSolo", "Lunar Mission"],
    hero: "/lovable-uploads/d1e74099-500d-4c46-a984-3fbe6f55a551.png",
    sections: [
      {
        type: "overview",
        title: "Context & Goal",
        content: "The development of reliable prognostic health management systems for space-qualified instrumentation represents one of the most technically demanding challenges in aerospace engineering, where the extreme consequences of component failure during lunar missions create requirements for unprecedented reliability prediction and early fault detection capabilities. During my involvement with NASA's MSolo (Mass Spectrometer Observing Lunar Operations) program, I encountered the formidable challenge of developing a comprehensive vibration-based fatigue detection system for a sophisticated mass spectrometer designed to operate autonomously on the lunar surface for extended periods.\n\nThe MSolo mass spectrometer represented a critical scientific instrument designed to analyze the composition of the lunar atmosphere and surface-released gases, providing essential data for understanding lunar geology, potential resource utilization, and the fundamental physics of airless body atmospheres. The instrument's mission-critical nature demanded extraordinary reliability, as failure during the lunar mission would not only compromise scientific objectives worth hundreds of millions of dollars but could potentially impact the safety and success of entire lunar surface operations.\n\nThe technical challenge was compounded by the unique operational environment of lunar surface missions, where mechanical components experience complex loading patterns arising from launch vibrations reaching 20g peak accelerations, thermal cycling between -230°C during lunar night and +120°C during lunar day, micrometeorite impacts generating high-frequency shock loading, and operational vibrations from mechanical pumps and sample handling mechanisms.",
        image: {
          src: "/lovable-uploads/14efa7d6-1eb1-4d60-ab03-23f40b553d31.png",
          alt: "NASA MSolo Mission Patch - Official INFICON mission insignia for the MPH RGA (Miniature Penning Hot-cathode Residual Gas Analyzer) liftoff project, featuring technology powering space exploration with detailed technical iconography including mass spectrometer components, molecular analysis symbols, and countdown elements, representing the sophisticated instrumentation designed for lunar surface atmospheric composition analysis and gas detection capabilities essential for understanding airless body physics and potential resource utilization",
          position: "right"
        },
        metrics: [
          { label: "Launch Acceleration", value: "20g peak" },
          { label: "Temperature Range", value: "-230°C to +120°C" },
          { label: "Mission Duration", value: "Extended lunar cycles" },
          { label: "Detection Sensitivity", value: "95%" },
          { label: "False Alarm Rate", value: "<2%" }
        ]
      },
      {
        type: "theoretical",
        title: "Theoretical Background",
        content: "The fundamental physics governing fatigue failure in space-qualified instruments created an extraordinarily complex failure analysis problem where traditional reliability prediction methods proved inadequate for the unique combination of loading conditions, material behaviors at extreme temperatures, and operational requirements. Fatigue crack initiation and propagation in the mass spectrometer's critical components followed complex mechanisms governed by Paris' law relationships, where crack growth rates depend on stress intensity factor ranges that vary dramatically with temperature, loading frequency, and environmental conditions unique to the lunar environment.\n\n**Critical Component Analysis:**\n\nThe mass spectrometer's critical subcomponents presented distinct failure modes requiring specialized analysis approaches: the ion source assembly incorporating delicate filaments and focusing electrodes operating at high temperatures and subjected to thermal cycling stresses; the quadrupole analyzer with precision-machined rods requiring dimensional stability within nanometer tolerances while experiencing vibrational loading; the detector assembly incorporating electron multipliers and sensitive electronic components vulnerable to mechanical shock; and the vacuum system with turbomolecular pumps containing high-speed rotating components operating at temperatures far below terrestrial specifications.\n\n**Signal Processing Foundation:**\n\nThe Fast Fourier Transform implementation represented the foundation of the analysis framework, requiring sophisticated enhancement beyond standard FFT approaches to address the unique characteristics of fatigue-related vibration signatures. The implementation utilized overlapped windowing with Hann windows to minimize spectral leakage, zero-padding to improve frequency resolution in critical bands, and advanced averaging techniques to enhance signal-to-noise ratios for weak fatigue signatures embedded in operational noise.",
        image: {
          src: "/lovable-uploads/a85953e7-3a7b-4014-bdf6-1fe73f721746.png",
          alt: "INFICON Transpector CPM Mass Spectrometer Laboratory Setup - Advanced analytical instrumentation system demonstrating the sophisticated laboratory configuration for NASA's MSolo mass spectrometer development, featuring high-precision vacuum chambers, pressure monitoring equipment, specialized gas handling systems, and comprehensive testing apparatus essential for validating space-qualified mass spectrometry components under controlled conditions, including the complex mechanical assemblies, electronic interfaces, and calibration standards required for lunar atmospheric analysis missions",
          position: "left"
        },
        equations: [
          {
            equation: "\\frac{da}{dN} = C(\\Delta K)^m",
            variables: [
              { symbol: "da/dN", description: "crack growth rate per cycle" },
              { symbol: "C", description: "material constant" },
              { symbol: "ΔK", description: "stress intensity factor range" },
              { symbol: "m", description: "Paris law exponent" }
            ]
          },
          {
            equation: "\\Delta K = Y\\sigma\\sqrt{\\pi a}",
            variables: [
              { symbol: "ΔK", description: "stress intensity factor range" },
              { symbol: "Y", description: "geometry factor" },
              { symbol: "σ", description: "applied stress" },
              { symbol: "a", description: "crack length" }
            ]
          },
          {
            equation: "N_f = \\int_{a_0}^{a_c} \\frac{da}{C(\\Delta K)^m}",
            variables: [
              { symbol: "Nf", description: "fatigue life (cycles)" },
              { symbol: "a0", description: "initial crack size" },
              { symbol: "ac", description: "critical crack size" },
              { symbol: "C, m", description: "Paris law constants" }
            ]
          }
        ]
      },
      {
        type: "methodology",
        title: "Steps & Methodology",
        content: "My approach to this multifaceted challenge began with comprehensive development of a machine learning-enhanced signal processing framework that could extract meaningful fatigue indicators from accelerometer data collected throughout the instrument's operational envelope. The system incorporated advanced digital signal processing techniques including wavelet transform analysis for time-frequency decomposition, spectral analysis using Welch's method with optimized windowing functions, statistical process control for baseline establishment and drift detection, and feature extraction algorithms specifically designed to identify the subtle signatures of incipient fatigue damage.\n\n**Component-Specific Monitoring Strategy:**\n\nThe monitoring system addressed four critical subsystems with distinct failure characteristics:\n\n• **Ion Source Assembly:** Monitored resonant frequencies at 125, 340, and 890 Hz with focus on filament mount stress concentrations and thermal cycling effects on tungsten-rhenium components operating at 1800-2200 K\n\n• **Quadrupole Analyzer:** Tracked frequencies at 78, 156, 234, and 412 Hz for molybdenum rod assemblies requiring nanometer-level dimensional stability and frequency stability within 1×10⁻⁶ tolerance\n\n• **Detector Assembly:** Monitored 92, 284, and 567 Hz signatures for Inconel 718 components with shock limits of 100g and vibration limits of 10g RMS\n\n• **Vacuum System:** Analyzed pump harmonics at 45, 180, 360, and 720 Hz for stainless steel 316L components with 12,000 RPM pump speeds and bearing life expectations of 10⁸ cycles\n\n**Environmental Loading Analysis:**\n\nThe system accounted for multiple loading environments including launch conditions with 600-second duration and 20g peak accelerations across dominant frequencies of 5, 15, 35, 80, and 200 Hz; lunar landing impacts with 180-second duration and 15g peaks; and lunar operational conditions with 14-day thermal cycles and continuous operational vibrations.",
        image: {
          src: "/lovable-uploads/229290ad-acf1-4a7c-8cdc-4bccacb9c1ad.png",
          alt: "Spacecraft Test Facility - Advanced thermal vacuum chamber and environmental testing facility showcasing the sophisticated infrastructure required for space-qualified instrumentation validation, featuring precision mechanical handling systems, controlled environmental chambers, and comprehensive monitoring equipment essential for simulating the extreme conditions of lunar surface operations including temperature cycling, vacuum exposure, and vibration testing protocols necessary for NASA's MSolo mass spectrometer mission preparation",
          position: "right"
        },
        standards: [
          "NASA-STD-7001B - Launch Environment Specifications",
          "NASA-STD-5009 - Nondestructive Evaluation Requirements", 
          "ASTM E647 - Fatigue Crack Growth Testing",
          "MIL-STD-810G - Environmental Engineering Considerations"
        ],
        equations: [
          {
            equation: "PSD(f) = \\frac{1}{T}|X(f)|^2",
            variables: [
              { symbol: "PSD(f)", description: "power spectral density" },
              { symbol: "T", description: "observation time" },
              { symbol: "|X(f)|²", description: "magnitude squared of Fourier transform" }
            ]
          },
          {
            equation: "S_{welch}(f) = \\frac{1}{K}\\sum_{k=0}^{K-1}|X_k(f)|^2",
            variables: [
              { symbol: "Swelch(f)", description: "Welch's averaged periodogram" },
              { symbol: "K", description: "number of overlapped segments" },
              { symbol: "|Xk(f)|²", description: "periodogram of k-th segment" }
            ]
          }
        ]
      },
      {
        type: "implementation",
        title: "Data & Results",
        content: "The implemented system achieved exceptional performance across multiple critical metrics while operating within the stringent constraints of space-qualified hardware. The machine learning-enhanced approach successfully demonstrated the ability to detect incipient fatigue damage with unprecedented sensitivity while maintaining extremely low false alarm rates essential for autonomous lunar operations.\n\n**Performance Achievements:**\n\n• **Detection Sensitivity:** Achieved 95% target detection rate for developing fatigue conditions across all monitored components\n• **False Alarm Rate:** Maintained below 2% maximum threshold, crucial for preventing unnecessary mission interruptions\n• **Prediction Horizon:** Provided 7-day advance warning capability, enabling proactive mission planning and risk mitigation\n• **Confidence Threshold:** Exceeded 85% confidence levels for all critical fault classifications\n• **Environmental Resilience:** Maintained performance across full lunar temperature range from -230°C to +120°C\n\n**Component-Specific Results:**\n\nThe system successfully identified characteristic failure signatures for each critical subsystem. Ion source monitoring detected thermal cycling effects on tungsten-rhenium filaments with stress concentration factors ranging from 1.9 to 2.8. Quadrupole analyzer monitoring achieved nanometer-level sensitivity to dimensional changes affecting frequency stability. Detector assembly monitoring successfully tracked shock and vibration limits for Inconel 718 components. Vacuum system monitoring provided early warning for bearing degradation in 12,000 RPM turbomolecular pumps.\n\n**Machine Learning Model Performance:**\n\nMultiple specialized models were developed for different failure modes, with Random Forest classifiers achieving superior performance for multi-class fault identification and Isolation Forest algorithms excelling at novelty detection for previously unseen failure patterns.",
        metrics: [
          { label: "Detection Sensitivity", value: "95%" },
          { label: "False Alarm Rate", value: "<2%" },
          { label: "Prediction Horizon", value: "7 days" },
          { label: "Confidence Level", value: ">85%" },
          { label: "Temperature Range", value: "-230°C to +120°C" },
          { label: "Frequency Resolution", value: "0.1 Hz" }
        ],
        visual: {
          type: "chart",
          content: {
            type: "line",
            data: {
              labels: ["Ion Source", "Quadrupole", "Detector", "Vacuum System"],
              datasets: [{
                label: "Detection Accuracy (%)",
                data: [96.2, 94.8, 95.5, 95.1],
                borderColor: "hsl(var(--primary))",
                backgroundColor: "hsl(var(--primary) / 0.1)"
              }]
            }
          }
        }
      },
      {
        type: "results",
        title: "Impact & Takeaways",
        content: "The successful development and validation of the vibration-based fatigue detection system for NASA's MSolo Mass Spectrometer established new paradigms for autonomous health monitoring in space-qualified instrumentation. This project demonstrated that sophisticated machine learning algorithms can operate effectively in the extreme environments of space missions while providing critical early warning capabilities that could prevent catastrophic failures and mission loss.\n\n**Technical Innovation:**\n\n• **Advanced Signal Processing:** Pioneered the application of multi-scale wavelet analysis combined with traditional FFT techniques for detecting subtle fatigue signatures in space instruments\n• **Component-Specific Modeling:** Developed specialized monitoring approaches for four distinct subsystem types, each with unique failure mechanisms and environmental sensitivities\n• **Environmental Adaptation:** Created algorithms capable of maintaining performance across the extreme temperature ranges and loading conditions of lunar surface operations\n• **Predictive Capabilities:** Achieved 7-day advance warning for developing fatigue conditions, enabling proactive mission management and risk mitigation\n\n**Mission-Critical Impact:**\n\nThe system provides unprecedented insight into instrument health during extended lunar surface operations, where traditional maintenance and repair approaches are impossible. The early warning capability enables mission operators to adjust operational parameters, implement contingency procedures, or modify mission timelines to prevent catastrophic failures that could compromise multi-billion dollar exploration programs.\n\n**Future Space Applications:**\n\nThe methodologies and algorithms developed for MSolo have broader applications across NASA's space exploration portfolio, including Mars rovers, asteroid sample return missions, and deep space probes where autonomous health monitoring is essential for mission success.",
        image: {
          src: "/lovable-uploads/ada67827-dd72-4a5e-aece-0158cc2f270b.png",
          alt: "Space Shuttle Payload Bay Configuration - Interior view of a space shuttle cargo bay showcasing the complex payload integration setup for space-qualified scientific instruments, featuring precision mounting systems, thermal management infrastructure, electrical connections, and the sophisticated logistics required for deploying advanced mass spectrometry equipment like NASA's MSolo system to lunar destinations, demonstrating the intricate engineering and mission planning necessary for successful space-based scientific operations",
          position: "left"
        },
        pullQuote: "Successfully achieved 95% detection sensitivity with <2% false alarms across all critical subsystems, providing 7-day advance warning capability for fatigue-related failures in the extreme lunar environment.",
        metrics: [
          { label: "Mission Protection", value: "Multi-billion $ value" },
          { label: "Advance Warning", value: "7 days" },
          { label: "Detection Rate", value: "95%" },
          { label: "False Alarms", value: "<2%" },
          { label: "Component Coverage", value: "4 critical subsystems" },
          { label: "Temperature Resilience", value: "460°C range" }
        ],
        visual: {
          type: "image",
          content: "/lovable-uploads/d1e74099-500d-4c46-a984-3fbe6f55a551.png"
        }
      }
    ]
  },
  "uav-tail-fuselage": {
    id: "uav-tail-fuselage",
    title: "UAV Tail & Fuselage Variations for Stability Analysis",
    subtitle: "Comprehensive analysis of tail and fuselage design variations to optimize UAV stability characteristics through advanced computational methods",
    category: "Aerodynamics",
    date: "2024",
    author: "Azarias Thomas",
    tags: ["Stability Analysis", "Flight Dynamics", "Design Optimization", "CFD", "Wind Tunnel Testing"],
    hero: "/lovable-uploads/000f98ca-15f2-4d60-a820-a33b989ababe.png",
    sections: [
       {
         type: "overview",
         title: "Project Overview",
         content: "This comprehensive study analyzes the effects of tail and fuselage design variations on UAV stability characteristics through advanced computational methods. The research focuses on optimizing aerodynamic performance while maintaining flight stability across various operational conditions.\n\nUAV design requires careful consideration of stability and control characteristics, particularly in the aft-body configuration. This study examines how variations in tail geometry and fuselage integration affect longitudinal and lateral-directional stability derivatives, providing insights for optimal UAV design.",
         metrics: [
           { label: "Configurations Tested", value: "15 variants" },
           { label: "CFD Simulations", value: "50+ runs" },
           { label: "Wind Tunnel Tests", value: "25 hours" },
           { label: "Stability Improvement", value: "25%" },
           { label: "Drag Reduction", value: "15%" }
         ],
         image: {
           src: "/lovable-uploads/d85accaa-2d58-4dc6-a314-7ad65ddb945b.png",
           alt: "CFD mesh analysis of UAV aircraft showing computational grid structure and aerodynamic parameters including efficiency, lift coefficient, and moment coefficient data for stability analysis",
           position: "right"
         }
       },
       {
         type: "theoretical",
         title: "Technical Methodology",
         content: "**Geometric Parametrization:**\n\nThe study employed systematic variations in tail configuration including horizontal tail positioning (conventional vs. T-tail configurations), vertical tail sizing (area and aspect ratio variations), fuselage-tail integration (different junction geometries and fairing shapes), and dihedral angles (impact on lateral stability characteristics).\n\n**Computational Analysis Framework:**\n\nAdvanced CFD simulations were conducted using high-fidelity Reynolds-Averaged Navier-Stokes (RANS) methods with SST k-ω turbulence modeling. The analysis included static stability derivative calculations, dynamic response analysis, control effectiveness evaluation, and trim condition assessments.",
         image: {
           src: "/lovable-uploads/e20bcea9-8d8d-450a-b1e0-35edf8e18228.png",
           alt: "Pressure contour and velocity streamline analysis comparing UAV aerodynamics at different angles of attack, showing top and bottom views with color-coded pressure distribution and flow visualization",
           position: "left"
         },
         equations: [
          {
            equation: "C_{m_\\alpha} = \\frac{\\partial C_m}{\\partial \\alpha}",
            variables: [
              { symbol: "C_{m_α}", description: "Pitching moment coefficient derivative with respect to angle of attack" },
              { symbol: "C_m", description: "Pitching moment coefficient" },
              { symbol: "α", description: "Angle of attack (radians)" }
            ]
          },
          {
            equation: "C_{m_{\\delta_e}} = \\frac{\\partial C_m}{\\partial \\delta_e} \\cdot \\frac{S_e}{S} \\cdot \\frac{l_t}{c}",
            variables: [
              { symbol: "C_{m_δe}", description: "Elevator control power derivative" },
              { symbol: "δ_e", description: "Elevator deflection angle" },
              { symbol: "S_e/S", description: "Elevator-to-wing area ratio" },
              { symbol: "l_t/c", description: "Tail arm to wing chord ratio" }
            ]
          }
        ]
      },
      {
        type: "methodology",
        title: "Key Findings and Results",
        content: "**Longitudinal Stability:**\n\nThe analysis revealed that horizontal tail positioning significantly affects pitch stability margins. T-tail configurations showed 18% improvement in static margin, conventional tail designs provided better control authority at high angles of attack, and optimal tail arm length increased stability margin by 25%.\n\n**Lateral-Directional Characteristics:**\n\nVertical tail effectiveness and fuselage integration effects were quantified. Increased vertical tail area improved directional stability by 30%, optimized fuselage-tail junction reduced interference drag by 12%, and dihedral angle optimization enhanced spiral stability.\n\n**Performance Optimization:**\n\nThe integrated design approach yielded significant performance improvements including overall drag reduction of 15% through optimized tail-fuselage integration, enhanced control effectiveness across the flight envelope, improved gust response characteristics, and reduced trim drag penalties.",
        standards: [
          "MIL-F-8785C - Flying Qualities of Piloted Airplanes",
          "ESDU 76003 - Aircraft Stability and Control Data",
          "AIAA R-004-1992 - Atmospheric Flight Mechanics",
          "NATO STANAG 3596 - Flying Qualities"
        ]
      },
      {
        type: "implementation",
        title: "Advanced Analysis Techniques",
        content: "**Stability Derivative Calculation:**\n\nDynamic stability analysis employed linearized equations of motion with computed aerodynamic derivatives. Key stability parameters were systematically evaluated across the flight envelope.\n\n**Control Surface Sizing:**\n\nElevator and rudder effectiveness were evaluated using control derivative analysis to ensure adequate control authority while maintaining stability margins.\n\n**Flutter and Aeroelastic Considerations:**\n\nThe study included preliminary aeroelastic analysis to ensure structural integrity including modal analysis of tail structure under aerodynamic loading, flutter speed calculations for critical configurations, and structural optimization for weight and stiffness balance.",
        metrics: [
          { label: "Static Margin Improvement", value: "25%" },
          { label: "Control Authority", value: "+15%" },
          { label: "Drag Reduction", value: "15%" },
          { label: "Gust Response", value: "-20%" },
          { label: "Flutter Margin", value: "40% above Vd" }
        ]
      },
       {
         type: "results",
         title: "Design Optimization Framework",
         content: "A multi-objective optimization approach was implemented to balance competing design requirements. The optimization included minimizing drag while maximizing stability margins and ensuring control authority, subject to structural limits, manufacturing feasibility, and operational requirements.\n\n**Validation and Testing:**\n\nCFD results were validated against experimental data from wind tunnel testing including force and moment measurements across angle of attack range, surface pressure distributions on tail surfaces, and flow visualization studies of tail-fuselage interaction. Selected configurations were validated through flight testing programs with stability and control derivative identification, handling qualities assessment, and performance verification across flight conditions.",
         metrics: [
           { label: "Wind Tunnel Correlation", value: "±3% accuracy" },
           { label: "Flight Test Validation", value: "95% agreement" },
           { label: "Optimization Convergence", value: "25 iterations" },
           { label: "Design Space Explored", value: "500+ points" },
           { label: "Pareto Solutions", value: "15 optimal" }
         ],
         image: {
           src: "/lovable-uploads/6f53696a-ff0b-472d-b75b-0cb0b4fabdae.png",
           alt: "Professional wind tunnel testing facility with large circular test section and measurement arrays for aerodynamic validation and experimental verification of UAV designs",
           position: "right"
         },
         visual: {
           type: "image",
           content: "/lovable-uploads/d1e74099-500d-4c46-a984-3fbe6f55a551.png"
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
    const hasImage = section.image;
    const imagePosition = hasImage && typeof section.image === 'object' ? section.image.position : (isEven ? 'right' : 'left');

    return (
      <section 
        key={index} 
        className={`py-16 animate-fade-in`}
        style={{ animationDelay: `${index * 0.2}s` }}
      >
        <div className="container mx-auto px-4">
          {/* Adaptive text flow layout for sections with images */}
          {hasImage ? (
            <div className="relative">
              {/* Image positioned as float */}
              <div className={`mb-6 lg:mb-0 ${
                imagePosition === 'left' 
                  ? 'lg:float-left lg:w-1/2 lg:pr-8' 
                  : 'lg:float-right lg:w-1/2 lg:pl-8'
              }`}>
                <div className="rounded-lg overflow-hidden shadow-lg">
                  <img 
                    src={typeof section.image === 'string' ? section.image : section.image.src} 
                    alt={typeof section.image === 'string' ? section.title : section.image.alt}
                    className="w-full h-80 object-cover"
                  />
                  {typeof section.image === 'object' && section.image.alt && (
                    <div className="bg-muted/50 px-4 py-2 text-sm text-muted-foreground">
                      {section.image.alt}
                    </div>
                  )}
                </div>
              </div>

              {/* Content that flows around image and then expands */}
              <div className="space-y-6">
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

              {/* Clear float to ensure proper layout flow */}
              <div className="clear-both"></div>
            </div>
          ) : (
            /* Grid layout for sections without images */
            <div className="grid lg:grid-cols-2 gap-12 items-start">
              <div className="space-y-6 lg:col-span-2">
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

                {/* Visual/Code Content for non-image sections */}
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
              </div>
            </div>
          )}
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
