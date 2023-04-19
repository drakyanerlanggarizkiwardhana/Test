returncode

    def call(self) -> int:
        """ Override for package installer specific logic.
        Returns
        -------
        int
            The return code of the package install process
        """
        raise NotImplementedError()

    def _non_gui_print(self, text: str, end: Optional[str] = None) -> None:
        """ Print output to console if not running in the GUI
        Parameters
        ----------
        text: str
            The text to print
        end: str, optional
            The line ending to use. Default: ``None`` (new line)
        """
        if self._is_gui:
            return
        print(text, end=end)

    def _seen_line_log(self, text: str) -> None:
        """ Output gets spammed to the log file when conda is waiting/processing. Only log each
        unique line once.
        Parameters
        ----------
        text: str
            The text to log
        """
        if text in self._seen_lines:
            return
        logger.verbose(text)  # type:ignore
        self._seen_lines.add(text)


class PexpectInstaller(Installer):  # pylint: disable=too-few-public-methods
    """ Package installer for Linux/macOS using Pexpect
    Uses Pexpect for installing packages allowing access to realtime feedback
    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    package: str
        The package name that is being installed
    command: list
        The command to run
    is_gui: bool
        ``True`` if the process is being called from the Faceswap GUI
    """
    def call(self) -> int:
        """ Install a package using the Pexpect module
        Returns
        -------
        int
            The return code of the package install process
        """
        import pexpect  # pylint:disable=import-outside-toplevel,import-error
        proc = pexpect.spawn(" ".join(self._command),
                             encoding=self._env.encoding, codec_errors="replace", timeout=None)
        while True:
            try:
                idx = proc.expect(["\r\n", "\r"])
                line = proc.before.rstrip()
                if line and idx == 0:
                    if self._last_line_cr:
                        self._last_line_cr = False
                        # Output last line of progress bar and go to next line
                        self._non_gui_print(line)
                    self._seen_line_log(line)
                elif line and idx == 1:
                    self._last_line_cr = True
                    logger.debug(line)
                    self._non_gui_print(line, end="\r")
            except pexpect.EOF:
                break
        proc.close()
        return proc.exitstatus


class WinPTYInstaller(Installer):  # pylint: disable=too-few-public-methods
    """ Package installer for Windows using WinPTY
    Spawns a pseudo PTY for installing packages allowing access to realtime feedback
    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    package: str
        The package name that is being installed
    command: list
        The command to run
    is_gui: bool
        ``True`` if the process is being called from the Faceswap GUI
    """
    def __init__(self,
                 environment: Environment,
                 package: str,
                 command: List[str],
                 is_gui: bool) -> None:
        super().__init__(environment, package, command, is_gui)
        self._cmd = which(command[0], path=os.environ.get('PATH', os.defpath))
        self._cmdline = list2cmdline(command)
        logger.debug("cmd: '%s', cmdline: '%s'", self._cmd, self._cmdline)

        self._pbar = re.compile(r"(?:eta\s[\d\W]+)|(?:\s+\|\s+\d+%)\Z")
        self._eof = False
        self._read_bytes = 1024

        self._lines: List[str] = []
        self._out = ""

    def _read_from_pty(self, proc: Any, winpty_error: Any) -> None:
        """ Read :attr:`_num_bytes` from WinPTY. If there is an error reading, recursively halve
        the number of bytes read until we get a succesful read. If we get down to 1 byte without a
        succesful read, assume we are at EOF.
        Parameters
        ----------
        proc: :class:`winpty.PTY`
            The WinPTY process
        winpty_error: :class:`winpty.WinptyError`
            The winpty error exception. Passed in as WinPTY is not in global scope
        """
        try:
            from_pty = proc.read(self._read_bytes)
        except winpty_error:
            # TODO Reinsert this check
            # The error message "pipe has been ended" is language specific so this check
            # fails on non english systems. For now we just swallow all errors until no
            # bytes are left to read and then check the return code
            # if any(val in str(err) for val in ["EOF", "pipe has been ended"]):
            #    # Get remaining bytes. On a comms error, the buffer remains unread so keep
            #    # halving buffer amount until down to 1 when we know we have everything
            #     if self._read_bytes == 1:
            #         self._eof = True
            #     from_pty = ""
            #     self._read_bytes //= 2
            # else:
            #     raise

            # Get remaining bytes. On a comms error, the buffer remains unread so keep
            # halving buffer amount until down to 1 when we know we have everything
            if self._read_bytes == 1:
                self._eof = True
            from_pty = ""
            self._read_bytes //= 2

        self._out += from_pty

    def _out_to_lines(self) -> None:
        """ Process the winpty output into separate lines. Roll over any semi-consumed lines to the
        next proc call. """
        if "\n" not in self._out:
            return

        self._lines.extend(self._out.split("\n"))

        if self._out.endswith("\n") or self._eof:  # Ends on newline or is EOF
            self._out = ""
        else:  # roll over semi-consumed line to next read
            self._out = self._lines[-1]
            self._lines = self._lines[:-1]

    def _parse_lines(self) -> None:
        """ Process the latest batch of lines that have been received from winPTY. """
        for line in self._lines:  # Dump the output to log
            line = line.rstrip()
            is_cr = bool(self._pbar.search(line))
            if line and not is_cr:
                if self._last_line_cr:
                    self._last_line_cr = False
                    if not self._env.is_installer:
                        # Go to next line
                        self._non_gui_print("")
                self._seen_line_log(line)
            elif line:
                self._last_line_cr = True
                logger.debug(line)
                # NSIS only updates on line endings, so force new line for installer
                self._non_gui_print(line, end=None if self._env.is_installer else "\r")
        self._lines = []

    def call(self) -> int:
        """ Install a package using the PyWinPTY module
        Returns
        -------
        int
            The return code of the package install process
        """
        import winpty  # pylint:disable=import-outside-toplevel,import-error
        # For some reason with WinPTY we need to pass in the full command. Probably a bug
        proc = winpty.PTY(
            80 if self._env.is_installer else 100,
            24,
            backend=winpty.enums.Backend.WinPTY,  # ConPTY hangs and has lots of Ansi Escapes
            agent_config=winpty.enums.AgentConfig.WINPTY_FLAG_PLAIN_OUTPUT)  # Strip all Ansi

        if not proc.spawn(self._cmd, cmdline=self._cmdline):
            del proc
            raise RuntimeError("Failed to spawn winpty")

        while True:
            self._read_from_pty(proc, winpty.WinptyError)
            self._out_to_lines()
            self._parse_lines()

            if self._eof:
                returncode = proc.get_exitstatus()
                break

        del proc
        return returncode


class SubProcInstaller(Installer):
    """ The fallback package installer if either of the OS specific installers fail.
    Uses the python Subprocess module to install packages. Feedback does not return in realtime
    so the process can look like it has hung to the end user
    Parameters
    ----------
    environment: :class:`Environment`
        Environment class holding information about the running system
    package: str
        The package name that is being installed
    command: list
        The command to run
    is_gui: bool
        ``True`` if the process is being called from the Faceswap GUI
    """
    def __init__(self,
                 environment: Environment,
                 package: str,
                 command: List[str],
                 is_gui: bool) -> None:
        super().__init__(environment, package, command, is_gui)
        self._shell = self._env.os_version[0] == "Windows" and command[0] == "conda"

    def __call__(self) -> int:
        """ Override default call function so we don't recursively call ourselves on failure. """
        returncode = self.call()
        logger.debug("Package: %s, returncode: %s", self._package, returncode)
        return returncode

    def call(self) -> int:
        """ Install a package using the Subprocess module
        Returns
        -------
        int
            The return code of the package install process
        """
        with Popen(self._command,
                   bufsize=0, stdout=PIPE, stderr=STDOUT, shell=self._shell) as proc:
            while True:
                if proc.stdout is not None:
                    line = proc.stdout.readline().decode(self._env.encoding, errors="replace")
                returncode = proc.poll()
                if line == "" and returncode is not None:
                    break

                is_cr = line.startswith("\r")
                line = line.rstrip()

                if line and not is_cr:
                    if self._last_line_cr:
                        self._last_line_cr = False
                        # Go to next line
                        self._non_gui_print("")
                    self._seen_line_log(line)
                elif line:
                    self._last_line_cr = True
                    logger.debug(line)
                    self._non_gui_print("", end="\r")
        return returncode


class Tips():
    """ Display installation Tips """
    @classmethod
    def docker_no_cuda(cls) -> None:
        """ Output Tips for Docker without Cuda """
        path = os.path.dirname(os.path.realpath(__file__))
        logger.info(
            "1. Install Docker\n"
            "https://www.docker.com/community-edition\n\n"
            "2. Build Docker Image For Faceswap\n"
            "docker build -t deepfakes-cpu -f Dockerfile.cpu .\n\n"
            "3. Mount faceswap volume and Run it\n"
            "# without GUI\n"
            "docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-cpu --name deepfakes-cpu \\ \n"
            "\t-v %s:/srv \\ \n"
            "\tdeepfakes-cpu\n\n"
            "# with gui. tools.py gui working.\n"
            "## enable local access to X11 server\n"
            "xhost +local:\n"
            "## create container\n"
            "nvidia-docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-cpu --name deepfakes-cpu \\ \n"
            "\t-v %s:/srv \\ \n"
            "\t-v /tmp/.X11-unix:/tmp/.X11-unix \\ \n"
            "\t-e DISPLAY=unix$DISPLAY \\ \n"
            "\t-e AUDIO_GID=`getent group audio | cut -d: -f3` \\ \n"
            "\t-e VIDEO_GID=`getent group video | cut -d: -f3` \\ \n"
            "\t-e GID=`id -g` \\ \n"
            "\t-e UID=`id -u` \\ \n"
            "\tdeepfakes-cpu \n\n"
            "4. Open a new terminal to run faceswap.py in /srv\n"
            "docker exec -it deepfakes-cpu bash", path, path)
        logger.info("That's all you need to do with a docker. Have fun.")

    @classmethod
    def docker_cuda(cls) -> None:
        """ Output Tips for Docker with Cuda"""
        path = os.path.dirname(os.path.realpath(__file__))
        logger.info(
            "1. Install Docker\n"
            "https://www.docker.com/community-edition\n\n"
            "2. Install latest CUDA\n"
            "CUDA: https://developer.nvidia.com/cuda-downloads\n\n"
            "3. Install Nvidia-Docker & Restart Docker Service\n"
            "https://github.com/NVIDIA/nvidia-docker\n\n"
            "4. Build Docker Image For Faceswap\n"
            "docker build -t deepfakes-gpu -f Dockerfile.gpu .\n\n"
            "5. Mount faceswap volume and Run it\n"
            "# without gui \n"
            "docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-gpu --name deepfakes-gpu \\ \n"
            "\t-v %s:/srv \\ \n"
            "\tdeepfakes-gpu\n\n"
            "# with gui.\n"
            "## enable local access to X11 server\n"
            "xhost +local:\n"
            "## enable nvidia device if working under bumblebee\n"
            "echo ON > /proc/acpi/bbswitch\n"
            "## create container\n"
            "nvidia-docker run -tid -p 8888:8888 \\ \n"
            "\t--hostname deepfakes-gpu --name deepfakes-gpu \\ \n"
            "\t-v %s:/srv \\ \n"
            "\t-v /tmp/.X11-unix:/tmp/.X11-unix \\ \n"
            "\t-e DISPLAY=unix$DISPLAY \\ \n"
            "\t-e AUDIO_GID=`getent group audio | cut -d: -f3` \\ \n"
            "\t-e VIDEO_GID=`getent group video | cut -d: -f3` \\ \n"
            "\t-e GID=`id -g` \\ \n"
            "\t-e UID=`id -u` \\ \n"
            "\tdeepfakes-gpu\n\n"
            "6. Open a new terminal to interact with the project\n"
            "docker exec deepfakes-gpu python /srv/faceswap.py gui\n",
            path, path)

    @classmethod
    def macos(cls) -> None:
        """ Output Tips for macOS"""
        logger.info(
            "setup.py does not directly support macOS. The following tips should help:\n\n"
            "1. Install system dependencies:\n"
            "XCode from the Apple Store\n"
            "XQuartz: https://www.xquartz.org/\n\n"

            "2a. It is recommended to use Anaconda for your Python Virtual Environment as this\n"
            "will handle the installation of CUDA and cuDNN for you:\n"
            "https://www.anaconda.com/distribution/\n\n"

            "2b. If you do not want to use Anaconda you will need to manually install CUDA and "
            "cuDNN:\n"
            "CUDA: https://developer.nvidia.com/cuda-downloads"
            "cuDNN: https://developer.nvidia.com/rdp/cudnn-download\n\n")

    @classmethod
    def pip(cls) -> None:
        """ Pip Tips """
        logger.info("1. Install PIP requirements\n"
                    "You may want to execute `chcp 65001` in cmd line\n"
                    "to fix Unicode issues on Windows when installing dependencies")


if __name__ == "__main__":
    logfile = os.path.join(os.path.dirname(os.path.realpath(sys.argv[0])), "faceswap_setup.log")
    log_setup("INFO", logfile, "setup")
    logger.debug("Setup called with args: %s", sys.argv)
    ENV = Environment()
    Checks(ENV)
    ENV.set_config()
    if _INSTALL_FAILED:
        sys.exit(1)
    Install(ENV)