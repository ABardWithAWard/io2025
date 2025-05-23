import { useEffect } from 'react';

function RedirectToDjangoAdmin() {
  useEffect(() => {
    window.location.href = '/admin/';
  }, []);

  return null;
}

export default RedirectToDjangoAdmin;