export const PRIVATEVARS = ['color', 'startPos', 'endPos', 'name', 'id', 'status'];  // USE THIS SO USERS CANT EDIT THESE VALUES
export const MODALPRIVATEVARS = [...PRIVATEVARS, 'fringeType'];
export const API_ROUTE = `http://localhost:${import.meta.env.VITE_BACKEND_API_PORT ?? 8000}`;
export const TWISS_OPTIONS = [
    { value: '\\epsilon (\\pi.mm.mrad)', label: '\\epsilon (\\pi.mm.mrad)', modal_val: 'emittance' },
    { value: '\\alpha', label: '\\alpha', modal_val: 'alpha' },
    { value: '\\beta (m)', label: '\\beta (m)', modal_val: 'beta' },
    { value: '\\gamma (rad/m)', label: '\\gamma (rad/m)', modal_val: 'gamma' },
    { value: 'D (mm)', label: 'D (mm)', modal_val: 'dispersion' },
    { value: 'D\' (mrad)', label: 'D\' (mrad)', modal_val: 'dispersion_prime' },
    { value: '\\phi (deg)', label: '\\phi (deg)', modal_val: 'angle' },
    { value: 'Envelope\\ E (mm)', label: 'Envelope\\ E (mm)', modal_val: 'envelope' }
]