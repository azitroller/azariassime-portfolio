import { useEffect, useRef, useState } from "react";
import { Mail, Linkedin, Twitter, Github } from "lucide-react";

const ContactSection = () => {
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

  const socialLinks = [
    {
      name: "Email",
      icon: Mail,
      href: "mailto:Azarias.sime@ufl.edu",
      label: "Azarias.sime@ufl.edu"
    },
    {
      name: "LinkedIn",
      icon: Linkedin,
      href: "https://linkedin.com/in/azarias-sime-01823624a",
      label: "linkedin.com/in/azarias-sime-01823624a"
    },
    {
      name: "Phone",
      icon: Twitter,
      href: "tel:+14157456161",
      label: "(415) 745-6161"
    },
    {
      name: "GitHub",
      icon: Github,
      href: "https://github.com/azarias-sime",
      label: "github.com/azarias-sime"
    }
  ];

  return (
    <section
      id="contact"
      ref={sectionRef}
      className="py-24 md:py-32 section-padding"
    >
      <div className="container-max">
        {/* Section Header */}
        <div className={`text-center mb-16 ${isVisible ? 'animate-fade-in' : 'opacity-0'}`}>
          <h2 className="text-sm font-medium text-primary mb-4 tracking-wide uppercase">
            Contact
          </h2>
          <h3 className="text-heading text-3xl md:text-4xl lg:text-5xl font-bold mb-8">
            Let's work together
          </h3>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            I'm always interested in new opportunities and collaborations. 
            Let's create something amazing together.
          </p>
        </div>

        {/* Contact Information - Full Width */}
        <div className={`max-w-4xl mx-auto ${isVisible ? 'animate-fade-in' : 'opacity-0'}`}>
          <div className="bg-card border border-border rounded-2xl p-12">
            <h4 className="text-3xl font-bold mb-8 text-foreground text-center">
              Get in touch
            </h4>
            
            <p className="text-muted-foreground mb-12 leading-relaxed text-center text-lg max-w-3xl mx-auto">
              I'm currently seeking internships and full-time opportunities in aerospace engineering and manufacturing. 
              Whether you have a project in mind or want to discuss collaboration, I'd love to connect.
            </p>

            <div className="grid md:grid-cols-2 gap-8">
              {socialLinks.map((social) => (
                <a
                  key={social.name}
                  href={social.href}
                  target={social.name !== "Email" ? "_blank" : undefined}
                  rel={social.name !== "Email" ? "noopener noreferrer" : undefined}
                  className="flex items-center gap-6 p-6 rounded-xl hover:bg-muted transition-all duration-300 group"
                >
                  <div className="w-16 h-16 bg-primary/10 rounded-xl flex items-center justify-center group-hover:bg-primary group-hover:text-white transition-all duration-300">
                    <social.icon className="w-6 h-6 text-primary group-hover:text-white" />
                  </div>
                  <div>
                    <h5 className="font-semibold text-foreground text-lg">{social.name}</h5>
                    <p className="text-muted-foreground">{social.label}</p>
                  </div>
                </a>
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ContactSection;