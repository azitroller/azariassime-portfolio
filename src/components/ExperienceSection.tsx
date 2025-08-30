import { useEffect, useRef, useState } from "react";
import { Calendar, MapPin, ExternalLink } from "lucide-react";

const ExperienceSection = () => {
  const [isVisible, setIsVisible] = useState(false);
  const sectionRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
        }
      },
      { threshold: 0.1 }
    );

    if (sectionRef.current) {
      observer.observe(sectionRef.current);
    }

    return () => observer.disconnect();
  }, []);

  const experiences = [
    {
      title: "Senior UX/UI Designer",
      company: "TechCorp Solutions",
      location: "Kuala Lumpur, Malaysia",
      period: "2022 - Present",
      description: "Leading design initiatives for fintech products, creating comprehensive design systems, and mentoring junior designers. Increased user engagement by 60% through data-driven design decisions.",
      achievements: [
        "Led redesign of core banking platform",
        "Established design system used across 5 products",
        "Mentored 3 junior designers"
      ]
    },
    {
      title: "UX/UI Designer",
      company: "Digital Innovations",
      location: "Kuala Lumpur, Malaysia",
      period: "2020 - 2022",
      description: "Designed user interfaces for web and mobile applications, conducted user research, and collaborated with cross-functional teams to deliver user-centered solutions.",
      achievements: [
        "Improved conversion rates by 35%",
        "Conducted 50+ user interviews",
        "Delivered 15+ product launches"
      ]
    },
    {
      title: "Junior UI Designer",
      company: "StartupHub",
      location: "Petaling Jaya, Malaysia",
      period: "2019 - 2020",
      description: "Created visual designs for startup clients, developed brand identities, and learned the fundamentals of user experience design in a fast-paced environment.",
      achievements: [
        "Designed for 20+ startup clients",
        "Created 10+ brand identities",
        "Mastered design fundamentals"
      ]
    }
  ];

  return (
    <section
      id="experience"
      ref={sectionRef}
      className="py-24 md:py-32 section-padding bg-secondary/10"
    >
      <div className="container-max">
        {/* Section Header */}
        <div className={`text-center mb-16 ${isVisible ? 'animate-fade-in' : 'opacity-0'}`}>
          <h2 className="text-sm font-medium text-primary mb-4 tracking-wide uppercase">
            Experience
          </h2>
          <h3 className="text-heading text-3xl md:text-4xl lg:text-5xl font-bold mb-8">
            Professional Journey
          </h3>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Over 5 years of experience crafting digital experiences that make a difference
          </p>
        </div>

        {/* Experience Timeline */}
        <div className="max-w-4xl mx-auto">
          {experiences.map((exp, index) => (
            <div
              key={exp.title + exp.company}
              className={`relative pl-8 pb-12 last:pb-0 ${
                isVisible ? 'animate-fade-in' : 'opacity-0'
              }`}
              style={{ animationDelay: `${index * 0.2}s` }}
            >
              {/* Timeline Line */}
              <div className="absolute left-0 top-0 w-px bg-border h-full">
                <div className="absolute -left-2 top-2 w-5 h-5 bg-primary rounded-full border-4 border-background shadow-glow" />
              </div>

              {/* Experience Card */}
              <div className="bg-card border border-border rounded-2xl p-8 hover:border-primary/30 transition-all duration-300 hover:shadow-soft">
                <div className="flex flex-col md:flex-row md:items-start justify-between mb-6">
                  <div className="mb-4 md:mb-0">
                    <h4 className="text-xl md:text-2xl font-bold text-foreground mb-2">
                      {exp.title}
                    </h4>
                    <h5 className="text-lg font-semibold text-primary mb-2">
                      {exp.company}
                    </h5>
                    <div className="flex flex-col sm:flex-row sm:items-center gap-4 text-sm text-muted-foreground">
                      <div className="flex items-center gap-2">
                        <Calendar className="w-4 h-4" />
                        {exp.period}
                      </div>
                      <div className="flex items-center gap-2">
                        <MapPin className="w-4 h-4" />
                        {exp.location}
                      </div>
                    </div>
                  </div>
                </div>

                <p className="text-muted-foreground leading-relaxed mb-6">
                  {exp.description}
                </p>

                <div>
                  <h6 className="font-semibold text-foreground mb-3">Key Achievements:</h6>
                  <ul className="space-y-2">
                    {exp.achievements.map((achievement, achIndex) => (
                      <li key={achIndex} className="flex items-start gap-3 text-muted-foreground">
                        <div className="w-2 h-2 bg-primary rounded-full mt-2 flex-shrink-0" />
                        {achievement}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Resume Download CTA */}
        <div className={`text-center mt-16 ${isVisible ? 'animate-fade-in' : 'opacity-0'}`}>
          <div className="bg-gradient-primary p-8 rounded-2xl text-center">
            <h4 className="text-2xl font-bold text-white mb-4">
              Want to know more about my experience?
            </h4>
            <p className="text-white/90 mb-6">
              Download my complete resume for detailed information about my skills and projects.
            </p>
            <button className="bg-white text-primary px-8 py-3 rounded-full font-semibold hover:bg-white/90 transition-all duration-300 inline-flex items-center gap-2">
              Download Resume
              <ExternalLink className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ExperienceSection;