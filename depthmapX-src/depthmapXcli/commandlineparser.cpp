// Copyright (C) 2017 Christian Sailer

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include "commandlineparser.h"
#include "exceptions.h"
#include "imodeparserfactory.h"
#include "parsingutils.h"
#include "version.h"
#include "gpu/host/accelcommon.h"
#include <iostream>
#include <cstring>
#include <algorithm>
#include <cstdlib>

using namespace depthmapX;

void CommandLineParser::printHelp(){
    std::cout << "Usage: depthmapXcli -m <mode> -f <filename> -o <output file> [-s] [-t <times.csv>] [-p] [mode options]\n"
              << "       depthmapXcli -v prints the current version\n"
              << "       depthmapXcli -h prints this help text\n"
              << "-s enables simple mode\n"
              << "-t <times.csv> enables output of runtimes as csv file\n"
              << "-p enables text progress printing\n"
#ifdef __APPLE__
              << "-gpu runs supported analyses on the GPU (Metal) when one is available,\n"
#else
              << "-gpu runs supported analyses on an NVIDIA GPU (CUDA) when one is available,\n"
#endif
              << "     otherwise on all CPU cores. Supported: VGA visibility (local, global)\n"
              << "-gpumin <points> smallest map sent to the GPU (default " << dmxaccel::kDefaultGpuMinNodes
              << ", 0 = always)\n"
              << "-serial runs VGA visibility, metric and angular with the original\n"
              << "     single-threaded code (default: all CPU cores)\n"

              << "Possible modes are:\n";
              std::for_each(_parserFactory.getModeParsers().begin(), _parserFactory.getModeParsers().end(), [](const ModeParserVec::value_type &p)->void{ std::cout << "  " << p->getModeName() << "\n"; });
              std::cout << "\n";
              std::for_each(_parserFactory.getModeParsers().begin(), _parserFactory.getModeParsers().end(), [](const ModeParserVec::value_type &p)->void{ std::cout << p->getHelp() << "\n"; });
              std::cout << std::flush;
}

void CommandLineParser::printVersion(){
    std::cout << TITLE_BASE << "\n" << std::flush;
}


CommandLineParser::CommandLineParser(const IModeParserFactory &parserFactory)
    :  m_simpleMode(false), m_printProgress(false), m_computeMode(Options::COMPUTE_CPU_PARALLEL),
       m_gpuMinNodes(dmxaccel::kDefaultGpuMinNodes), _parserFactory(parserFactory), _modeParser(0)
{}

void CommandLineParser::parse(size_t argc, char *argv[])
{
    m_valid = false;
    m_printVersionMode = false;
    if (argc <= 1)
    {
        throw CommandLineException("No commandline parameters provided - don't know what to do");
    }
    for ( size_t i = 1; i < argc;  )
    {
        if ( std::strcmp("-h", argv[i])== 0)
        {
            return;
        }
        else if ( std::strcmp("-v", argv[i])== 0)
        {
            m_printVersionMode = true;
            return;
        }
        else if ( std::strcmp ("-m", argv[i]) == 0)
        {
            if ( _modeParser)
            {
                throw CommandLineException("-m can only be used once");
            }
            ENFORCE_ARGUMENT("-m", i)

            for (auto iter = _parserFactory.getModeParsers().begin(), end = _parserFactory.getModeParsers().end();
                 iter != end; ++iter )
            {
                if ((*iter)->getModeName() == argv[i])
                {
                    _modeParser = iter->get();
                    break;
                }
            }

            if (!_modeParser)
            {
                throw CommandLineException(std::string("Invalid mode: ") + argv[i]);
            }
        }
        else if ( std::strcmp ("-f", argv[i]) == 0)
        {
            ENFORCE_ARGUMENT("-f", i)
            m_fileName = argv[i];
        }
        else if ( std::strcmp ("-o", argv[i]) == 0)
        {
            ENFORCE_ARGUMENT("-o", i)
            m_outputFile = argv[i];
        }
        else if ( std::strcmp ("-t", argv[i]) == 0)
        {
            ENFORCE_ARGUMENT("-t", i)
            m_timingFile = argv[i];
        }
        else if ( std::strcmp("-s", argv[i]) == 0)
        {
            m_simpleMode = true;
        }
        else if ( std::strcmp("-p", argv[i]) == 0)
        {
            m_printProgress = true;
        }
        else if ( std::strcmp("-gpu", argv[i]) == 0)
        {
            if (m_computeMode == Options::COMPUTE_SERIAL)
            {
                throw CommandLineException("-gpu and -serial cannot be used together");
            }
            m_computeMode = Options::COMPUTE_GPU;
        }
        else if ( std::strcmp("-serial", argv[i]) == 0)
        {
            if (m_computeMode == Options::COMPUTE_GPU)
            {
                throw CommandLineException("-gpu and -serial cannot be used together");
            }
            m_computeMode = Options::COMPUTE_SERIAL;
        }
        else if ( std::strcmp("-gpumin", argv[i]) == 0)
        {
            ENFORCE_ARGUMENT("-gpumin", i)
            char *end = nullptr;
            long value = std::strtol(argv[i], &end, 10);
            if (end == argv[i] || *end != '\0' || value < 0 || value > 2000000000L)
            {
                throw CommandLineException(std::string("-gpumin must be a non-negative integer, got ") + argv[i]);
            }
            m_gpuMinNodes = int(value);
        }
        ++i;
    }

    if (!_modeParser)
    {
        throw CommandLineException("-m for mode is required");
    }
    if (m_fileName.empty())
    {
        throw CommandLineException("-f for input file is required");
    }
    if (m_outputFile.empty())
    {
        throw CommandLineException("-o for output file is required");
    }
    _modeParser->parse(argc, argv);
    m_valid = true;
}

void CommandLineParser::run(IPerformanceSink &perfWriter) const
{
    if (!m_valid || !_modeParser)
    {
        throw CommandLineException("Trying to run with invalid command line parameters");
    }
    _modeParser->run(*this, perfWriter);
}

