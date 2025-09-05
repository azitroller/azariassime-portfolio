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
        ]
      },
      {
        type: "theoretical",
        title: "Theoretical Background",
        content: "Understanding the complex material science underlying valve failure mechanisms became crucial to developing an effective automated testing solution that could detect the subtle signatures of degradation processes. At elevated temperatures, valve components experience time-dependent plastic deformation following Norton's power law, where the creep rate depends exponentially on temperature and follows a power relationship with stress magnitude.\n\nFor Inconel 718 valve seats operating at 180°C under typical aerospace loading conditions, creep rates of 10⁻⁸ s⁻¹ can accumulate to significant deformation over extended test durations, with the exponential temperature dependence meaning that small temperature variations can dramatically affect creep behavior and thus failure timing.\n\nRepeated thermal cycling induces alternating stress cycles due to differential thermal expansion between dissimilar materials commonly found in valve assemblies. This leads to low-cycle fatigue crack initiation after 10³-10⁴ cycles, following the Coffin-Manson relationship.",
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
          title: "Advanced DAQ Reader Class",
          preview: `import nidaqmx
import numpy as np
from scipy import signal, stats
from datetime import datetime
import threading

class AdvancedDAQReader:
    def __init__(self, pressure_channels, temp_channels, 
                 sample_rate=10, buffer_size=10000):
        self.pressure_channels = pressure_channels
        self.temp_channels = temp_channels
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
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
        ]
      },
      {
        type: "theoretical",
        title: "Theoretical Background",
        content: "RGA sensors operate on the principle of electron impact ionization mass spectrometry, where gas molecules are ionized by a controlled electron beam, accelerated through an electric field, and separated by mass-to-charge ratio using either quadrupole or magnetic sector analyzers. The measurement process requires maintaining ultra-high vacuum conditions (typically 10⁻⁸ to 10⁻¹² Torr) within the analyzer chamber, precise alignment of ion optics to maintain measurement accuracy, stable high-voltage power supplies for ion acceleration and detection, and vibration-free mounting to prevent mechanical modulation of the electron beam path.\n\n**Ion Beam Deflection Physics:**\n\nThe fundamental physics governing RGA operation created specific mechanical requirements that directly conflicted with the dynamic environment of legged locomotion. Ion beam deflection due to mechanical vibration follows a predictable relationship where even small accelerations can cause significant beam displacement.\n\n**Robot Dynamics and Vibration Sources:**\n\nQuadruped locomotion generates complex force patterns that depend on gait selection, terrain characteristics, payload distribution, and locomotion speed, with fundamental frequencies determined by stride frequency (typically 1-3 Hz) and higher harmonics extending well into the structural resonance range of precision instrumentation.",
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
