import { useContext, useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";
import { signOut } from "firebase/auth";
import { auth } from "../firebase";
import { AuthContext } from "../auth/AuthContext";
import { NAV_LINKS } from "../utils/nav";

export default function Navbar() {
  const { user } = useContext(AuthContext);
  const navigate = useNavigate();
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const handleSignOut = async () => {
    await signOut(auth);
    navigate("/login");
  };

  const initial = user?.displayName
    ? user.displayName.charAt(0).toUpperCase()
    : "?";

  return (
    <header className="fixed top-4 left-1/2 -translate-x-1/2 z-50">
      <nav
        className="rounded-full bg-slate-800/90 backdrop-blur border border-slate-700 shadow-xl px-6 py-3 flex gap-6 items-center"
        aria-label="Main navigation"
      >
        {NAV_LINKS.map(({ to, label }) => (
          <NavLink
            key={to}
            to={to}
            className={({ isActive }) =>
              isActive
                ? "text-sky-400 font-semibold text-sm"
                : "text-slate-400 hover:text-slate-100 text-sm transition-colors"
            }
          >
            {label}
          </NavLink>
        ))}

        {user && (
          <div className="relative ml-auto">
            <button
              type="button"
              onClick={() => setDropdownOpen((prev) => !prev)}
              className="w-8 h-8 rounded-full bg-sky-600 flex items-center justify-center text-white text-sm font-semibold"
              aria-label="User menu"
            >
              {initial}
            </button>

            {dropdownOpen && (
              <div className="bg-slate-800 border border-slate-700 rounded-lg shadow-lg p-2 absolute right-0 top-full mt-2 z-50">
                <button
                  type="button"
                  onClick={handleSignOut}
                  className="text-sm text-red-400 hover:text-red-300 px-3 py-1.5 rounded cursor-pointer whitespace-nowrap"
                >
                  Sign out
                </button>
              </div>
            )}
          </div>
        )}
      </nav>
    </header>
  );
}
