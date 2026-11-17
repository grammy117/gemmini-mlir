"""Check physical group DMA coverage, tails, budgets and tamper refusal."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'compiler_optimized'),str(REPO/'compiler_optimized/compiler_library')]
from backend.shared_products import checked_schedule,_epoch_index_kernel,_literal_epoch
from backend.shared_rhs_transfers import select_rhs_block_transfers
from backend.shared_readout_configs import select_raw_readout_configuration
from merlin.sched.ir import Call,Const,Loop


def prepare(specs):
    schedule=checked_schedule(specs)
    epoch,_=_epoch_index_kernel(schedule.kernel)
    return select_rhs_block_transfers(schedule,epoch,tuple(row[2] for row in specs))


class TransferChecks(unittest.TestCase):
    def test_three_members_preserve_all_bands_with_fewer_calls(self):
        selected,r=prepare(((8,32,32),)*3)
        selected.revalidate()
        self.assertEqual((r['original_physical_rhs_commands'],r['physical_rhs_commands'],r['logical_rhs_bands']),(12,6,12))

    def test_two_wide_members(self):
        selected,r=prepare(((8,32,64),)*2)
        selected.revalidate()
        self.assertEqual((r['original_physical_rhs_commands'],r['physical_rhs_commands']),(16,4))

    def test_different_member_pitches(self):
        selected,r=prepare(((5,32,32),(5,32,64)))
        selected.revalidate()
        self.assertEqual((r['original_physical_rhs_commands'],r['physical_rhs_commands']),(12,4))

    def test_partial_column_tails_retain_checked_original(self):
        selected,r=prepare(((5,19,17),(5,19,33)))
        self.assertIsNone(selected)
        self.assertFalse(r['applied'])

    def test_changed_actual_width_refuses(self):
        selected,_=prepare(((8,32,32),)*3)
        changed=False
        def body(items):
            nonlocal changed
            result=[]
            for item in items:
                if isinstance(item,Loop):item=replace(item,body=body(item.body))
                elif isinstance(item,Call) and 'mvin_weight' in item.instr and not changed:
                    item=replace(item,args=tuple((k,Const(48) if k=='cols' else v) for k,v in item.args))
                    changed=True
                result.append(item)
            return tuple(result)
        bad=replace(selected,kernel=replace(selected.kernel,body=body(selected.kernel.body)))
        self.assertTrue(changed)
        with self.assertRaises(ValueError):bad.revalidate()

    def test_large_epoch_keeps_counted_loops(self):
        selected,_=prepare(((8,32,64),)*2)
        kernel,r=_literal_epoch(selected.kernel,max_calls=1)
        self.assertIs(kernel,selected.kernel)
        self.assertFalse(r['applied'])

    def test_uniform_readout_revalidates_complete_store_stream(self):
        schedule=checked_schedule(((8,32,32),)*3)
        epoch,_=_epoch_index_kernel(schedule.kernel,readout_epochs=True)
        wide,_=select_rhs_block_transfers(schedule,epoch,(32,32,32))
        configured=select_raw_readout_configuration(wide.kernel,32)
        configured.revalidate()
        self.assertTrue(configured.report['data_and_completion_unchanged'])
        # Losing a real completion edge must invalidate the final program.
        altered=replace(configured,kernel=replace(configured.kernel,body=configured.kernel.body[:-1]))
        with self.assertRaises(ValueError):altered.revalidate()


if __name__=='__main__':unittest.main()
