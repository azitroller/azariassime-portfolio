import { useNavigate, useLocation } from "react-router-dom";

const Footer = () => {
  const currentYear = new Date().getFullYear();
  const navigate = useNavigate();
  const location = useLocation();

  const handleHomeClick = () => {
    if (location.pathname !== '/') {
      navigate('/');
      setTimeout(() => window.scrollTo({ top: 0, behavior: 'smooth' }), 100);
    } else {
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  return (
    <footer className="py-12 section-padding border-t border-border">
      <div className="container-max">
        {/* Disclaimer */}
        <div className="mb-8 p-6 bg-muted/50 rounded-lg border border-border">
          <p className="text-sm text-muted-foreground leading-relaxed">
            <strong className="text-foreground">Disclaimer:</strong> The technical specifications, proprietary data, quantitative results, methodologies, and specific job duties presented in this portfolio have been omitted, modified, or generalized due to intellectual property restrictions and non-disclosure agreement obligations. Project images, descriptions, technologies used, and general outcomes are presented for portfolio demonstration purposes. All work samples comply with confidentiality requirements while showcasing relevant skills and experience.
          </p>
        </div>

        <div className="flex flex-col md:flex-row items-center justify-between gap-6">
          {/* Logo and Copyright */}
          <div className="flex items-center gap-8">
            <button 
              onClick={handleHomeClick}
              className="text-2xl font-bold text-foreground hover:text-primary transition-colors duration-300"
            >
              Azarias.
            </button>
            <p className="text-sm text-muted-foreground">
              © {currentYear} Azarias Sime. All rights reserved.
            </p>
          </div>

          {/* Contact Link */}
          <div className="flex items-center gap-6">
            <a
              href="mailto:Azarias.sime@ufl.edu"
              className="text-muted-foreground hover:text-primary transition-colors duration-300"
            >
              Azarias.sime@ufl.edu
            </a>
            <button
              onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
              className="text-sm text-muted-foreground hover:text-primary transition-colors duration-300"
            >
              Back to top
            </button>
          </div>
        </div>
      </div>
    </footer>
  );
};

export default Footer;