export const PRIVATEVARS = ['color', 'startPos', 'endPos', 'name', 'id', 'status'];  // USE THIS SO USERS CANT EDIT THESE VALUES
export const MODALPRIVATEVARS = [...PRIVATEVARS, 'fringeType'];
export const API_ROUTE = `http://localhost:${import.meta.env.VITE_BACKEND_API_PORT ?? 8000}`;